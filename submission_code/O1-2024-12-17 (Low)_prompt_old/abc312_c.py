def solve():
    import sys
    import bisect

    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    A = list(map(int, data[2:2+N]))
    B = list(map(int, data[2+N:]))

    A.sort()
    B.sort()

    # S(x) = number of sellers who can sell at or below x = count of A_i <= x
    # We can find it via bisect_right(A, x)
    def sellers(x):
        return bisect.bisect_right(A, x)

    # B(x) = number of buyers who can buy at or above x = count of B_i >= x
    # We can find it via M - bisect_left(B, x)
    def buyers(x):
        return M - bisect.bisect_left(B, x)

    # Binary Search for the minimal integer X where S(X) >= B(X)
    left, right = 1, 10**9 + 2  # upper bound a bit larger than max possible B_i

    while left < right:
        mid = (left + right) // 2
        if sellers(mid) >= buyers(mid):
            right = mid
        else:
            left = mid + 1

    print(left)

# Uncomment the following line if you need to run it interactively
# solve()