import sys
import threading

def main():
    import sys
    import bisect

    input = sys.stdin.readline
    N, T = map(int, input().split())
    S = input().strip()
    X = list(map(int, input().split()))

    # Collect positions of ants facing left ('0') for j>i queries.
    zeros = [X[i] for i in range(N) if S[i] == '0']
    zeros.sort()
    M = len(zeros)
    # BIT over zeros array
    bit = [0] * (M + 1)
    def bit_add(i, v):
        while i <= M:
            bit[i] += v
            i += i & -i
    def bit_sum(i):
        s = 0
        while i > 0:
            s += bit[i]
            i -= i & -i
        return s
    # initialize frequencies
    for i in range(1, M+1):
        bit_add(i, 1)

    ans = 0
    twoT = 2 * T

    for i in range(N):
        xi = X[i]
        if S[i] == '0':
            # remove this zero-ant from BIT
            pos = bisect.bisect_left(zeros, xi)
            # pos is 0-based, BIT is 1-based
            bit_add(pos+1, -1)
        else:
            # count zeros at positions xj where xi < xj <= xi + 2T
            # find first index with zeros[idx] > xi
            left = bisect.bisect_right(zeros, xi)
            # find last index with zeros[idx] <= xi + 2T
            right = bisect.bisect_right(zeros, xi + twoT)
            if left < right:
                # BIT sum from left+1 to right
                cnt = bit_sum(right) - bit_sum(left)
                ans += cnt

    print(ans)

if __name__ == "__main__":
    main()