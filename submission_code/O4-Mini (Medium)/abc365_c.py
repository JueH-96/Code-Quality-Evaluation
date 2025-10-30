import sys
import threading

def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    A = list(map(int, it))
    # Total sum of A
    total = sum(A)
    # If we can afford full costs, x can be infinite
    if total <= M:
        print("infinite")
        return
    # Sort A and build prefix sums
    A.sort()
    prefix = [0] * (N + 1)
    for i in range(N):
        prefix[i+1] = prefix[i] + A[i]
    import bisect
    
    # Function to compute f(x) = sum(min(x, A[i]))
    def f(x):
        # all A[i] <= x contribute A[i], others contribute x
        idx = bisect.bisect_right(A, x)
        # sum of A[0:idx]
        s_small = prefix[idx]
        # remaining N - idx elements each contribute x
        s_big = (N - idx) * x
        return s_small + s_big
    
    # Binary search for the maximum x in [0, max(A)]
    lo, hi = 0, A[-1]
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if f(mid) <= M:
            lo = mid
        else:
            hi = mid - 1
    print(lo)

if __name__ == "__main__":
    main()