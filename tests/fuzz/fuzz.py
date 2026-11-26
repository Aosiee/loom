#!/usr/bin/env python3
"""Runs loom_fuzz over many seeds in parallel, and shrinks a failing program.

  fuzz.py run BIN --seeds 1-1000 [--size 20] [--grow] [--jobs 8] [--timeout 120] [--out DIR]
      Runs seeds in parallel. --grow cycles the size through 10, 40, 100, 200, 400
      (seed s gets sizes[s % 5]). A run fails on a non-zero exit, a FUZZ line,
      a sanitizer report or a loom drain report; its log goes to DIR/fail-<seed>.log.

  fuzz.py shrink BIN (--seed S --size N | --program TEXT) [--match REGEX] [--tries 3]
      Removes epochs and operations and shrinks sizes and pool settings while
      the failure (REGEX, by default the first failure line's kind) still
      happens in one of --tries runs. Prints the smallest program found.

  Both take --extra="FLAGS" for every run (--extra=--block-only adds
  BlockOnly waits, which can hang without loom being wrong; see fuzz.cpp).
"""
import argparse
import concurrent.futures as cf
import os
import re
import subprocess
import sys
import time

BAD = re.compile(r"FUZZ (CHECK FAILED|HANG|FAILED)|ERROR: (Address|Thread|Leak|Undefined)Sanitizer|WARNING: ThreadSanitizer|"
                 r"runtime error:|did not finish|Assertion .* failed|loom: fatal|terminate called|Segmentation|Aborted")


EXTRA = []  # --extra: flags passed to every run (--block-only, say)


def run_one(binary, args, timeout):
    env = dict(os.environ)
    supp = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "tsan.supp")
    env.setdefault("TSAN_OPTIONS", "halt_on_error=1 second_deadlock_stack=1 suppressions=" + os.path.normpath(supp))
    env.setdefault("ASAN_OPTIONS", "detect_leaks=1")
    env.setdefault("UBSAN_OPTIONS", "print_stacktrace=1 halt_on_error=1")
    t0 = time.time()
    try:
        p = subprocess.run([binary] + args + EXTRA + ["--timeout", str(timeout)], capture_output=True, text=True,
                           timeout=timeout + 60, env=env)
        out, rc = p.stdout + p.stderr, p.returncode
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        out += "\nFUZZ HANG (driver timeout)\n"
        rc = -999
    return rc, out, time.time() - t0


def failed(rc, out):
    return rc != 0 or BAD.search(out) is not None


def signature(out):
    m = re.search(r"FUZZ CHECK FAILED \(fuzz.cpp:(\d+)\)", out)
    if m:
        return r"FUZZ CHECK FAILED \(fuzz.cpp:%s\)" % m.group(1)
    for pat in [r"FUZZ HANG", r"ERROR: AddressSanitizer: [\w-]+", r"WARNING: ThreadSanitizer: [\w ]+",
                r"runtime error:", r"Assertion .* failed", r"did not finish", r"terminate called"]:
        m = re.search(pat, out)
        if m:
            return re.escape(m.group(0))
    return r"."


def cmd_run(a):
    lo, hi = (int(x) for x in a.seeds.split("-"))
    sizes = [10, 40, 100, 200, 400]
    os.makedirs(a.out, exist_ok=True)
    fails, done, cpu = 0, 0, 0.0
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        futs = {}
        for s in range(lo, hi + 1):
            size = sizes[s % len(sizes)] if a.grow else a.size
            futs[ex.submit(run_one, a.binary, ["--seed", str(s), "--size", str(size)], a.timeout)] = (s, size)
        for f in cf.as_completed(futs):
            s, size = futs[f]
            rc, out, took = f.result()
            done += 1
            cpu += took
            if failed(rc, out):
                fails += 1
                with open(os.path.join(a.out, "fail-%d.log" % s), "w") as fh:
                    fh.write("seed %d size %d rc %d\n%s" % (s, size, rc, out))
                print("FAIL seed %d size %d rc %d: %s" % (s, size, rc, signature(out)), flush=True)
    print("%s: %d runs, %d failed, %.0f s of run time" % (a.binary, done, fails, cpu), flush=True)
    return 1 if fails else 0


def parse(text):
    epochs = []
    for ep in text.strip().split(";"):
        if not ep:
            continue
        head, _, ops = ep.partition(":")
        fields = head[1:].split(",")
        epochs.append([fields, [o for o in ops.split("/") if o]])
    return epochs


def unparse(epochs):
    return "".join("E" + ",".join(f) + ":" + "".join(o + "/" for o in ops) + ";" for f, ops in epochs)


def cmd_shrink(a):
    if a.program:
        text = a.program
    else:
        text = subprocess.run([a.binary, "--seed", str(a.seed), "--size", str(a.size), "--print"], capture_output=True,
                              text=True).stdout.strip()
    rc, out, _ = run_one(a.binary, ["--program", text], a.timeout)
    tries = 1
    while not failed(rc, out) and tries < a.tries * 3:
        rc, out, _ = run_one(a.binary, ["--program", text], a.timeout)
        tries += 1
    if not failed(rc, out):
        print("the program does not fail", file=sys.stderr)
        return 2
    match = re.compile(a.match or signature(out))
    print("matching /%s/" % match.pattern, file=sys.stderr)

    def fails(t):
        for _ in range(a.tries):
            rc, out, _ = run_one(a.binary, ["--program", t], a.timeout)
            if failed(rc, out) and match.search(out):
                return True
        return False

    best = parse(text)
    progress = True
    while progress:
        progress = False
        # Whole epochs, then chunks of operations (largest first), then single ones.
        for i in range(len(best)):
            if len(best) > 1:
                cand = best[:i] + best[i + 1:]
                if fails(unparse(cand)):
                    best, progress = cand, True
                    print("dropped epoch -> %s" % unparse(best), file=sys.stderr)
                    break
        for ei in range(len(best)):
            ops = best[ei][1]
            chunk = max(1, len(ops) // 2)
            while chunk >= 1:
                i = 0
                while i < len(best[ei][1]):
                    ops = best[ei][1]
                    cand = [list(e) for e in best]
                    cand[ei] = [best[ei][0], ops[:i] + ops[i + chunk:]]
                    if cand[ei][1] and fails(unparse(cand)):
                        best, progress = cand, True
                        print("dropped %d ops -> %s" % (chunk, unparse(best)), file=sys.stderr)
                    else:
                        i += chunk
                chunk //= 2
            # Smaller sizes, joins at once.
            for oi in range(len(best[ei][1])):
                kind, seed, size, join = best[ei][1][oi].split(".")
                for new in ([kind, seed, str(int(size) // 2), join] if int(size) > 1 else None,
                            [kind, seed, size, "0"] if join != "0" else None):
                    if not new:
                        continue
                    cand = [[list(e[0]), list(e[1])] for e in best]
                    cand[ei][1][oi] = ".".join(new)
                    if fails(unparse(cand)):
                        best, progress = cand, True
                        print("op %d -> %s" % (oi, cand[ei][1][oi]), file=sys.stderr)
            # Pool settings: workers, dedicated threads, limiters, drain.
            for fi, smaller in ((0, lambda v: str(int(v) // 2)), (4, lambda v: str(int(v) - 1)), (5, lambda v: str(max(2, int(v) - 1))),
                                (6, lambda v: "0")):
                v = best[ei][0][fi]
                nv = smaller(v)
                if nv != v and int(nv) >= 0:
                    cand = [[list(e[0]), list(e[1])] for e in best]
                    cand[ei][0][fi] = nv
                    if fails(unparse(cand)):
                        best, progress = cand, True
                        print("epoch field %d -> %s" % (fi, nv), file=sys.stderr)
    print(unparse(best))
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("binary")
    r.add_argument("--seeds", default="1-100")
    r.add_argument("--size", type=int, default=20)
    r.add_argument("--grow", action="store_true")
    r.add_argument("--jobs", type=int, default=8)
    r.add_argument("--timeout", type=int, default=120)
    r.add_argument("--out", default="fuzz-out")
    s = sub.add_parser("shrink")
    s.add_argument("binary")
    s.add_argument("--seed", type=int)
    s.add_argument("--size", type=int, default=20)
    s.add_argument("--program")
    s.add_argument("--match")
    s.add_argument("--tries", type=int, default=3)
    s.add_argument("--timeout", type=int, default=60)
    for p in (r, s):
        p.add_argument("--extra", default="", help="flags for every run, e.g. --extra=--block-only")
    a = ap.parse_args()
    EXTRA.extend(a.extra.split())
    sys.exit(cmd_run(a) if a.cmd == "run" else cmd_shrink(a))


if __name__ == "__main__":
    main()
