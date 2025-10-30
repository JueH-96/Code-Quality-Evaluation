import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(10**7)
    input = sys.stdin.readline

    N, Q = map(int, input().split())
    S = input().strip()

    # pre1[i]: number of '1' in S[0:i]  (i chars)
    pre1 = [0] * (N+1)
    for i, ch in enumerate(S, start=1):
        pre1[i] = pre1[i-1] + (1 if ch=='1' else 0)

    # suf2[i]: number of '2' in S[i-1:N] (from position i to end)
    suf2 = [0] * (N+2)
    for i in range(N, 0, -1):
        suf2[i] = suf2[i+1] + (1 if S[i-1]=='2' else 0)

    # collect slash positions (1-based)
    slashes = []
    for i,ch in enumerate(S, start=1):
        if ch=='/':
            slashes.append(i)

    M = len(slashes)
    # Build arrays x and y over slashes
    # x[j] = pre1[ slash_pos[j] -1 ]
    # y[j] = suf2[ slash_pos[j] +1 ]
    x = [0]*M
    y = [0]*M
    for j,pos in enumerate(slashes):
        x[j] = pre1[pos-1]
        y[j] = suf2[pos+1]

    import bisect

    out = []
    for _ in range(Q):
        L, R = map(int, input().split())
        # find the slice of slashes within [L,R]
        lo = bisect.bisect_left(slashes, L)
        hi = bisect.bisect_right(slashes, R) - 1
        if lo>hi:
            out.append('0')
            continue
        # constants
        a = pre1[L-1]
        b = suf2[R+1]
        # f(j) = min(x[j]-a, y[j]-b)
        # we want max f(j) over j in [lo..hi]
        # find first j in [lo..hi] with x[j]-a >= y[j]-b
        Lb, Hb = lo, hi+1
        while Lb < Hb:
            mid = (Lb+Hb)//2
            if x[mid] - a >= y[mid] - b:
                Hb = mid
            else:
                Lb = mid+1
        # consider candidates Lb and Lb-1
        best = 0
        for j in (Lb-1, Lb):
            if lo <= j <= hi:
                t = min(x[j]-a, y[j]-b)
                if t>best:
                    best = t
        if best < 0:
            best = 0
        ans = 2*best + 1
        out.append(str(ans))

    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()