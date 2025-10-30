import sys
import threading
def main():
    import sys
    data = sys.stdin.read().split()
    t = int(data[0])
    idx = 1
    out = []
    INF = 10**9
    for _ in range(t):
        n = int(data[idx]); idx+=1
        P = list(map(int, data[idx:idx+n]))
        idx += n
        # Check if already identity
        already = True
        for i, v in enumerate(P, start=1):
            if v != i:
                already = False
                break
        if already:
            out.append("0")
            continue
        # Build prefix max
        pref_max = [0]*(n+1)
        for i in range(1, n+1):
            pref_max[i] = max(pref_max[i-1], P[i-1])
        # Build suffix min
        suff_min = [INF]*(n+2)
        for i in range(n, 0, -1):
            suff_min[i] = min(suff_min[i+1], P[i-1])
        # Try find a pivot k for a single operation
        found = False
        for k in range(1, n+1):
            # need P[k]==k
            if P[k-1] != k:
                continue
            # prefix 1..k-1 must all be <= k-1
            if pref_max[k-1] > k-1:
                continue
            # suffix k+1..n must all be >= k+1
            if suff_min[k+1] < k+1:
                continue
            found = True
            break
        out.append("1" if found else "2")
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()