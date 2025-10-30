import sys
import bisect

def main():
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split())) # Seller prices (A_i or more)
    B = list(map(int, sys.stdin.readline().split())) # Buyer prices (B_i or less)

    A.sort()
    B.sort()

    # Binary search for the minimum X satisfying the condition.
    # The condition is: num_sellers(X) >= num_buyers(X).
    
    # Define the search range for X.
    # Smallest possible X is 1.
    # Largest possible X for the answer is 10^9 + 1.
    # (At X = 10^9 + 1, num_sellers = N, num_buyers = 0, so N >= 0 holds).
    b_s_low = 1
    b_s_high = 10**9 + 1   
    ans = b_s_high + 1     # Initialize `ans` to a value greater than any possible answer.

    while b_s_low <= b_s_high:
        mid = b_s_low + (b_s_high - b_s_low) // 2
        
        # Calculate num_sellers(mid): count of A_k <= mid
        num_s = bisect.bisect_right(A, mid)
        
        # Calculate num_buyers(mid): count of B_k >= mid
        # This is M - (count of B_k < mid)
        # count of B_k < mid is bisect.bisect_left(B, mid)
        num_b = M - bisect.bisect_left(B, mid)
        
        if num_s >= num_b:  # Condition met: mid is a potential answer
            ans = mid       # Record mid as a candidate
            b_s_high = mid - 1 # Try to find an even smaller X
        else: # Condition not met: mid is too small
            b_s_low = mid + 1
            
    print(ans)

if __name__ == '__main__':
    main()