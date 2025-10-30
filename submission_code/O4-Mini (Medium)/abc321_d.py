import sys
import threading

def main():
    import sys
    import bisect

    data = sys.stdin.buffer.read().split()
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    P = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    B = [int(next(it)) for _ in range(M)]
    B.sort()
    # build prefix sums for B
    prefix = [0] * (M + 1)
    for i in range(M):
        prefix[i + 1] = prefix[i] + B[i]

    total = 0
    for a in A:
        # for each a, find count of b where a+b <= P -> b <= P-a
        x = P - a
        k = bisect.bisect_right(B, x)
        # sum of min(a+b, P) for these k is k*a + sum(B[0..k-1])
        total += k * a + prefix[k]
        # for the remaining M-k pairs, cost is P each
        total += (M - k) * P

    print(total)

if __name__ == "__main__":
    main()