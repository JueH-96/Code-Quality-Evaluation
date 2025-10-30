def solve():
    import sys
    import math

    data = sys.stdin.read().split()
    N = int(data[0])
    M = int(data[1])
    K = int(data[2])

    # Compute LCM using GCD
    gcd_nm = math.gcd(N, M)
    lcm_nm = N // gcd_nm * M

    # Function f(x) = count of positive integers <= x
    # that are divisible by exactly one of N or M
    def count_exactly_one(x):
        return (x // N) + (x // M) - 2 * (x // lcm_nm)

    # Binary search for the smallest x such that f(x) >= K
    left, right = 1, 2 * max(N, M) * K  # upper bound to ensure we cover the K-th number
    while left < right:
        mid = (left + right) // 2
        if count_exactly_one(mid) >= K:
            right = mid
        else:
            left = mid + 1

    print(left)

def main():
    solve()

if __name__ == "__main__":
    main()