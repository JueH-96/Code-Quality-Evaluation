import sys
import threading

def main():
    import sys
    data = sys.stdin
    readline = data.readline

    T_line = readline().strip()
    if not T_line:
        return
    T = int(T_line)
    out = []
    for _ in range(T):
        line = readline().strip()
        while line == "":
            # skip blank lines
            line = readline().strip()
        n = int(line)
        A = list(map(int, readline().split()))
        # compute prefix sums and total
        S = 0
        ok = True
        # we will accumulate prefix sum as we go
        # check for k = 1..n-1:
        #   L = ceil(P_k / k)
        #   R = floor((S_tot - P_k) / (n - k))
        # if L > R for any k => impossible
        # else possible
        P = 0
        # We'll read all A, compute S
        S = sum(A)
        # Now walk prefixes
        P = 0
        # We can stop early if fail
        for k in range(1, n):
            P += A[k-1]
            # lower bound
            # Ceil(P/k) = (P + k-1)//k
            L = (P + k - 1) // k
            # upper bound
            rem = S - P
            cnt = n - k
            # floor(rem / cnt)
            R = rem // cnt
            if L > R:
                ok = False
                # no need to check further
                break
        out.append("Yes" if ok else "No")
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()