import sys
import threading

def main():
    import sys
    from bisect import bisect_right

    data = sys.stdin.read().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    S = sum(A)
    low = S // N
    c_high = S % N
    high = low + 1

    A.sort()
    # Build prefix sums
    P = [0] * (N + 1)
    for i in range(N):
        P[i+1] = P[i] + A[i]

    # Group1: targets = low, size = N - c_high
    size1 = N - c_high
    sum1 = 0
    if size1 > 0:
        # find first index in [0..size1-1] where A[i] > low
        p = bisect_right(A, low, 0, size1)
        cnt1 = size1 - p
        if cnt1 > 0:
            sum1 = (P[size1] - P[p]) - low * cnt1

    # Group2: targets = high, size = c_high
    sum2 = 0
    if c_high > 0:
        s = N - c_high
        # find first index where A[i] > high over whole array
        up = bisect_right(A, high)
        # but only in group2 (i >= s)
        q = up if up >= s else s
        cnt2 = N - q
        if cnt2 > 0:
            sum2 = (P[N] - P[q]) - high * cnt2

    ans = sum1 + sum2
    print(ans)

if __name__ == "__main__":
    main()