def solve():
    import sys
    data = sys.stdin.read().strip().split()
    A, M, L, R = map(int, data)
    
    # We want to count integers k such that L <= A + k*M <= R.
    # That is equivalent to L - A <= k*M <= R - A.
    # So (L - A)/M <= k <= (R - A)/M.
    #
    # Because Python's integer division floors the result, we use the identity:
    #   ceil(x) = - floor(-x)
    #
    # Hence:
    #   k_min = ceil((L - A) / M) = -floor(-(L - A)/M)
    #   k_max = floor((R - A) / M)
    #
    # Once we have k_min and k_max (both integers), the count is:
    #   max(0, k_max - k_min + 1).

    def ceil_div(n, d):
        # Returns ceil(n / d) for d > 0, using only integer arithmetic.
        return - ( -n // d )

    k_min = ceil_div(L - A, M)
    k_max = (R - A) // M  # floor division is already floor

    result = max(0, k_max - k_min + 1)
    print(result)

# Let's call solve() for completeness
def main():
    solve()

if __name__ == "__main__":
    main()