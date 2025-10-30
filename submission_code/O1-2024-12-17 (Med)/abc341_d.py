def main():
    import sys
    import math

    data = sys.stdin.read().strip().split()
    N, M, K = map(int, data)

    # Ensure N < M for convenience (not strictly necessary, but can simplify some logic).
    # If N > M, just swap them.
    if N > M:
        N, M = M, N

    # Compute LCM using GCD
    gcd_nm = math.gcd(N, M)
    lcm_nm = (N // gcd_nm) * M  # avoids potential overflow, though Python can handle big int anyway

    # f(x) = number of integers <= x that are divisible by exactly one of N or M
    def count_exactly_one(x):
        return (x // N) + (x // M) - 2 * (x // lcm_nm)

    # Binary search for the smallest x such that count_exactly_one(x) == K
    left, right = 1, K * M  # An upper bound that is definitely large enough
    while left < right:
        mid = (left + right) // 2
        if count_exactly_one(mid) >= K:
            right = mid
        else:
            left = mid + 1

    print(left)

# Do not forget to call main()!
main()