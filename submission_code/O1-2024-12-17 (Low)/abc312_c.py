def main():
    import sys
    import bisect

    input_data = sys.stdin.read().strip().split()
    N, M = map(int, input_data[:2])
    A = list(map(int, input_data[2:2+N]))
    B = list(map(int, input_data[2+N:2+N+M]))

    A.sort()
    B.sort()

    # We want the smallest X such that:
    #   (# of sellers with A_i <= X) >= (# of buyers with B_i >= X).

    # Define a function to check the condition for a given x
    def can(x):
        # Sellers who can sell at x => A_i <= x
        sell_count = bisect.bisect_right(A, x)
        # Buyers who can buy at x => B_i >= x
        # => M - index of first B_i >= x
        buy_count = M - bisect.bisect_left(B, x)
        return sell_count >= buy_count

    # We'll do binary search over [1, 10^9 + 1]
    left, right = 1, 10**9 + 1
    while left < right:
        mid = (left + right) // 2
        if can(mid):
            right = mid
        else:
            left = mid + 1

    print(left)

# Do not forget to call main()
if __name__ == "__main__":
    main()