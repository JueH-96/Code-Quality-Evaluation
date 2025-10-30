import sys
import bisect

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    P = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    B = [int(next(it)) for _ in range(M)]
    B.sort()
    # build prefix sums of B
    prefix = [0] * (M + 1)
    for i in range(M):
        prefix[i + 1] = prefix[i] + B[i]
    ans = 0
    for a in A:
        # For this a, find how many B[j] satisfy a + B[j] <= P
        x = P - a
        # pos = number of B[j] <= x
        pos = bisect.bisect_right(B, x)
        # sum of min(a+B[j], P) for j < pos is sum(a+B[j])
        sum1 = a * pos + prefix[pos]
        # for the rest j >= pos, min(a+B[j], P) = P
        sum2 = (M - pos) * P
        ans += sum1 + sum2
    print(ans)

if __name__ == '__main__':
    main()