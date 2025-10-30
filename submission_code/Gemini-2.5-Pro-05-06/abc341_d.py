import math

def solve():
    N, M, K = map(int, input().split())

    common_divisor = math.gcd(N, M)
    # LCM = (N * M) / GCD(N, M)
    # Python's integers handle arbitrary size, so N * M itself won't overflow.
    L = (N * M) // common_divisor

    # Function to count numbers <= x_val that are divisible by exactly one of N or M
    def count_numbers_le_x(x_val):
        if x_val == 0: # Base case, though binary search range starts from 1.
            return 0
        
        count_n = x_val // N
        count_m = x_val // M
        count_lcm = x_val // L
        
        # Count of numbers divisible by exactly one of N or M is:
        # (count_n) + (count_m) - 2 * (count_lcm)
        return count_n + count_m - 2 * count_lcm

    # Binary search for the K-th number
    low = 1
    # Upper bound estimation:
    # Max K is 10^10. Max N, M is 10^8.
    # Worst-case scenarios suggest answers around 10^18 to 2*10^18.
    # 4*10^18 provides a safe margin.
    high = 4 * (10**18) 
    ans = high # Initialize ans with a value that will be updated

    while low <= high:
        mid = low + (high - low) // 2 # Standard way to calculate mid to avoid overflow
                                      # (though not an issue for Python's default int size)
        
        num_found = count_numbers_le_x(mid)
        
        if num_found >= K:
            # mid is a potential answer. We want the smallest such mid,
            # so we try smaller values.
            ans = mid
            high = mid - 1
        else: # num_found < K
            # mid is too small. The K-th number must be larger.
            low = mid + 1
            
    print(ans)

if __name__ == '__main__':
    solve()