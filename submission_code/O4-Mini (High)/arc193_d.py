import sys
import threading
def main():
    import sys
    data = sys.stdin
    readline = data.readline
    write = sys.stdout.write
    # Read number of test cases
    line = readline().strip()
    if not line:
        return
    T = int(line)
    for _ in range(T):
        line = readline().strip()
        while line == "":
            line = readline().strip()
        N = int(line)
        A = readline().strip()
        B = readline().strip()
        # Collect positions of '1's (0-based)
        Apos = [i for i, ch in enumerate(A) if ch == '1']
        Bpos = [i for i, ch in enumerate(B) if ch == '1']
        nA = len(Apos)
        nB = len(Bpos)
        # If fewer initial pieces than needed targets, impossible
        if nA < nB:
            write("-1
")
            continue
        # We will test feasibility by greedy scan with a cost threshold x
        # Define a helper to test adjacency-only feasibility (x large)
        # But we inline the check into binary search, so first do one check at maxX
        maxX = 2 * N + 5
        # Inline ok(maxX) for adjacency feasibility
        Ap = Apos
        Bp = Bpos
        # First, test adjacency (cost constraint always true for x = maxX)
        i = -1
        prev_a = 0  # previous matched A position
        feasible = True
        for k in range(nB):
            if k == 0:
                needed = -10**18  # no adjacency requirement
            else:
                # need Apos[i] >= prev_a + (Bpos[k] - Bpos[k-1])
                needed = prev_a + (Bp[k] - Bp[k-1])
            # move i forward to satisfy adjacency (cost always <= maxX)
            i += 1
            # find i such that Ap[i] >= needed
            # Since cost constraint is huge, we accept first Ap[i] >= needed
            # We may need to skip multiple until condition met
            while i < nA and Ap[i] < needed:
                i += 1
            if i >= nA:
                feasible = False
                break
            prev_a = Ap[i]
        if not feasible:
            write("-1
")
            continue
        # Now do binary search on minimal x
        low = -1
        high = maxX
        # We'll inline ok(x) check
        while high - low > 1:
            mid = (low + high) // 2
            # check(mid):
            i = -1
            prev_a = 0
            ok = True
            # We scan through Bpos targets k=0..nB-1, matching to Apos
            for k in range(nB):
                if k == 0:
                    needed = -10**18
                else:
                    needed = prev_a + (Bp[k] - Bp[k-1])
                # find next Apos index i > prev_i satisfying adjacency and cost
                i += 1
                # scan until we hit condition or run out
                while i < nA:
                    a = Ap[i]
                    # adjacency constraint
                    if a < needed:
                        i += 1
                        continue
                    # cost = abs(a - Bp[k]) + (i - k)
                    diff = a - Bp[k]
                    if diff < 0:
                        diff = -diff
                    # i - k >= 0 always since i >= k
                    w = diff + (i - k)
                    if w <= mid:
                        # match Bpos[k] to Ap[i]
                        prev_a = a
                        break
                    # else try next i
                    i += 1
                else:
                    # no break => no i found
                    ok = False
                if not ok:
                    break
            if ok:
                high = mid
            else:
                low = mid
        write(str(high) + "
")

if __name__ == "__main__":
    main()