def main():
    import sys
    import math
    
    data = sys.stdin.read().strip().split()
    N, M, K = map(int, data)
    
    # Compute gcd and lcm of N and M
    g = math.gcd(N, M)
    L = (N // g) * M  # lcm
    
    # Count how many numbers ≤ x are divisible by exactly one of N or M
    def count_exactly_one(x):
        return x // N + x // M - 2 * (x // L)
    
    # Binary search for the smallest number x such that
    # there are at least K numbers ≤ x divisible by exactly one of N or M.
    lo, hi = 1, max(N, M) * K
    
    while lo < hi:
        mid = (lo + hi) // 2
        if count_exactly_one(mid) < K:
            lo = mid + 1
        else:
            hi = mid
    
    # lo is the K-th smallest positive integer divisible by exactly one of N or M
    print(lo)

# Do not forget to call main
if __name__ == '__main__':
    main()