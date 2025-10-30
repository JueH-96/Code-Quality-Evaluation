import sys
import threading

def main():
    import sys
    from bisect import bisect_left, bisect_right
    
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    X = [int(next(it)) for _ in range(N)]
    P = [int(next(it)) for _ in range(N)]
    # build prefix sums of P
    prefix = [0] * (N + 1)
    for i in range(N):
        prefix[i+1] = prefix[i] + P[i]
    
    Q = int(next(it))
    out = []
    for _ in range(Q):
        L = int(next(it))
        R = int(next(it))
        # find first index with X[idx] >= L
        li = bisect_left(X, L)
        # find last index with X[idx] <= R => bisect_right-1
        ri = bisect_right(X, R) - 1
        if li <= ri:
            total = prefix[ri+1] - prefix[li]
        else:
            total = 0
        out.append(str(total))
    
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()