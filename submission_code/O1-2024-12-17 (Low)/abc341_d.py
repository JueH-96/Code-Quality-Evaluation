def main():
    import sys
    from math import gcd
    
    data = sys.stdin.read().strip().split()
    N, M, K = map(int, data)
    
    # Compute LCM using GCD
    lcm = N * M // gcd(N, M)
    
    # We define f(x) = number of integers <= x divisible by exactly one of N or M
    # f(x) = (x//N) + (x//M) - 2*(x//lcm)
    def count_exactly_one(x):
        return (x // N) + (x // M) - 2 * (x // lcm)
    
    # Binary search for the smallest x such that f(x) >= K
    left, right = 1, 2 * 10**18
    while left < right:
        mid = (left + right) // 2
        if count_exactly_one(mid) >= K:
            right = mid
        else:
            left = mid + 1
    
    # left will be the K-th smallest integer divisible by exactly one of N and M
    print(left)

# Call main() to run the solution
if __name__ == "__main__":
    main()