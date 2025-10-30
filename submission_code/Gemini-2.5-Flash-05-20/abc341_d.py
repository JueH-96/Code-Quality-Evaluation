import math

def solve():
    N, M, K = map(int, input().split())

    # Calculate LCM (Least Common Multiple)
    # lcm(a, b) = (a * b) // gcd(a, b)
    # Python's math.gcd works for positive integers.
    g = math.gcd(N, M)
    L = (N * M) // g

    # Define the count function:
    # count_exact_one(limit) returns the number of positive integers up to 'limit'
    # that are divisible by exactly one of N and M.
    def count_exact_one(limit):
        # Number of multiples of N up to limit: limit // N
        # Number of multiples of M up to limit: limit // M
        # Number of multiples of both N and M (i.e., multiples of LCM) up to limit: limit // L
        
        # Numbers divisible by N only = (multiples of N) - (multiples of LCM)
        # Numbers divisible by M only = (multiples of M) - (multiples of LCM)
        
        # Total count = (multiples of N only) + (multiples of M only)
        #             = (limit // N - limit // L) + (limit // M - limit // L)
        #             = (limit // N) + (limit // M) - 2 * (limit // L)
        return (limit // N) + (limit // M) - 2 * (limit // L)

    # Binary search for the K-th number
    low = 1
    # The maximum possible value for the K-th number.
    # If N or M is 1, count_exact_one(X) is roughly X. So, max X could be K (10^10).
    # If N, M are large (e.g., 10^8), count_exact_one(X) is roughly X / min(N, M).
    # So X could be roughly K * min(N, M) = 10^10 * 10^8 = 10^18.
    # A safe upper bound is 2 * 10^18. Python integers handle arbitrary size.
    high = 2 * 10**18 
    
    ans = high # Initialize ans to a value that will be updated by binary search

    while low <= high:
        mid = (low + high) // 2
        
        current_count = count_exact_one(mid)
        
        if current_count >= K:
            # If the count up to 'mid' is K or more, 'mid' is a possible answer.
            # We store it and try to find a smaller possible answer in the lower half.
            ans = mid
            high = mid - 1
        else:
            # If the count up to 'mid' is less than K, 'mid' is too small.
            # We need to look for a larger number in the upper half.
            low = mid + 1
            
    print(ans)

# Ensure the solve function is called when the script is executed
if __name__ == '__main__':
    solve()