def main():
    import sys
    import bisect
    input = sys.stdin.readline

    # Read input
    N, M = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))

    # Sort seller and buyer price lists
    A.sort()
    B.sort()

    # We'll search for X in [1, max(A_i, B_i) + 1]
    left = 1
    maxAB = max(A[-1], B[-1])
    right = maxAB + 1

    # Binary search for the smallest X such that
    # (# of sellers with A_i <= X) >= (# of buyers with B_i >= X).
    while left < right:
        mid = (left + right) // 2
        # Sellers willing to sell at price mid or less
        s_count = bisect.bisect_right(A, mid)
        # Buyers willing to pay at price mid or more
        t_count = M - bisect.bisect_left(B, mid)

        if s_count >= t_count:
            right = mid
        else:
            left = mid + 1

    # left == right is the minimum X satisfying the condition
    print(left)

if __name__ == "__main__":
    main()