import sys
import threading

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    t = int(next(it))
    out_lines = []
    for _ in range(t):
        n = int(next(it))
        k = int(next(it))
        # Read array a
        a = [int(next(it)) for _ in range(n)]
        # Handle k = 4 separately, since it's 2^2
        if k == 4:
            # Compute total 2-adic valuation S = sum v2(a_i)
            S = 0
            for ai in a:
                v = 0
                x = ai
                # count factors of 2
                while (x & 1) == 0:
                    v += 1
                    x >>= 1
                S += v
            if S >= 2:
                out_lines.append("0")
            elif S == 1:
                # We need one more factor of 2; one increment on an odd a_i suffices
                out_lines.append("1")
            else:
                # S == 0: need two factors of 2
                # Option A: flip two odds -> cost = 2
                # Option B: for some ai, add until ai %4 == 0, cost = (4 - ai%4)%4
                # Take min of these
                min_c2 = 10**9
                for ai in a:
                    c2 = (4 - (ai & 3)) & 3
                    if c2 < min_c2:
                        min_c2 = c2
                # best is min(2, min_c2)
                ans = 2 if 2 < min_c2 else min_c2
                out_lines.append(str(ans))
        else:
            # k is 2,3, or 5 (all prime)
            # minimal cost to make some a_i divisible by k
            best = k  # upper bound
            for ai in a:
                c = (k - (ai % k)) % k
                if c < best:
                    best = c
                    if best == 0:
                        break
            out_lines.append(str(best))
    sys.stdout.write("
".join(out_lines))

if __name__ == "__main__":
    main()