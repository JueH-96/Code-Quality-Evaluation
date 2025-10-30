import sys
import threading
def main():
    input = sys.stdin.readline
    T = int(input())
    out = []
    for _ in range(T):
        n = int(input())
        A = input().strip()
        B = input().strip()
        S = [i for i,ch in enumerate(A,1) if ch=='1']
        Tpos = [i for i,ch in enumerate(B,1) if ch=='1']
        M = len(S)
        K = len(Tpos)
        if M < K:
            out.append("-1")
            continue
        t0 = Tpos[0]
        t1 = Tpos[-1]
        best = 10**18
        # sliding window of length K over S
        # we need S[i] <= t0 and S[i+K-1] >= t1
        # ops = max(t1 - S[i], S[i+K-1] - t0)
        for i in range(M - K + 1):
            lo = S[i]
            hi = S[i+K-1]
            if lo <= t0 and hi >= t1:
                ops = max(t1 - lo, hi - t0)
                if ops < best:
                    best = ops
        if best == 10**18:
            out.append("-1")
        else:
            out.append(str(best))
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()