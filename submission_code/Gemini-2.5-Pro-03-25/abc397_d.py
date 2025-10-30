# YOUR CODE HERE
import math
import sys

# Using binary search to find integer square root
def integer_sqrt(n):
    """
    Computes the integer square root floor(sqrt(n)) using binary search.
    Returns -1 for negative input. Assumes n fits within standard integer types.
    Python handles large integers automatically.
    """
    if n < 0:
        # This case should not be reached based on the calling context,
        # but included for robustness.
        return -1 
    if n == 0:
        return 0
    
    # Determine a reasonable upper bound for binary search.
    # The value s = sqrt(S) where S = (4M - k^2)/3.
    # Max S is approx 4*N/3. For N=10^18, max S ~ 1.34 * 10^18.
    # The max square root s is approx sqrt(1.34 * 10^18) ~ 1.16 * 10^9.
    # Using 2 * 10^9 as the upper bound for binary search is safe.
    low = 0
    # Set high slightly larger than max possible s to cover edge cases.
    high = 2 * 10**9 + 1 

    best_s = 0
    while low <= high:
        mid = (low + high) // 2
        
        # Handle mid=0 case properly. If n > 0, sqrt(n) >= 1.
        if mid == 0:
           # If n=0, it's handled by the initial check.
           # If n > 0, the square root must be at least 1.
           # Ensure search proceeds in the positive range.
           low = 1 
           continue

        # Calculate mid_squared using Python's arbitrary precision integers.
        mid_squared = mid * mid
        
        if mid_squared == n:
            return mid # Exact square root found
        elif mid_squared < n:
            # mid is a potential candidate for floor(sqrt(n))
            # Store it as the best candidate so far and try larger values.
            best_s = mid 
            low = mid + 1
        else: # mid_squared > n
            # mid is too large. Reduce the upper bound.
            high = mid - 1
            
    # After the loop, best_s holds the largest integer s such that s*s <= n.
    # This is floor(sqrt(n)).
    return best_s

def is_perfect_square(num):
    """
    Checks if a non-negative integer is a perfect square using integer_sqrt.
    Returns (True, sqrt(num)) if it is a perfect square, (False, -1) otherwise.
    """
    # The context ensures num >= 0. Check defensively.
    if num < 0: return False, -1 
    if num == 0: return True, 0
    
    # Compute the integer square root (floor value).
    s = integer_sqrt(num)
    
    # Check if the square of the computed root equals the original number.
    if s * s == num:
        return True, s
    else:
        # If s*s != num, then num is not a perfect square.
        return False, -1

def solve():
    # Read the input integer N
    N = int(sys.stdin.readline())

    found = False

    # The problem asks for positive integers (x, y) such that x^3 - y^3 = N.
    # This implies x > y. Let x = y + k, where k is a positive integer (k >= 1).
    # Substituting x = y+k into the equation gives k(3y^2 + 3yk + k^2) = N.
    # Let M = N/k. Then 3y^2 + 3ky + k^2 = M.
    # This is a quadratic equation in y: 3y^2 + 3ky + (k^2 - M) = 0.
    # For integer solutions y, the discriminant Delta = 12M - 3k^2 must be a perfect square.
    # Let Delta = D^2. D^2 = 3(4M - k^2).
    # This implies 4M - k^2 >= 0, which means k^3 <= 4N.
    # Also, D^2 must be divisible by 9, so 4M - k^2 must be divisible by 3.
    # Let S = (4M - k^2) / 3. Then D^2 = 9S. This requires S to be a perfect square, say S = s^2.
    # The solutions for y are y = (-3k +/- D) / 6 = (-3k +/- 3s) / 6 = (s - k)/2 or (-s - k)/2.
    # Since y must be positive, y = (s - k) / 2 > 0. This requires s > k.
    # For y to be an integer, s-k must be even, meaning s and k must have the same parity.
    # If these conditions are met, x = y+k = (s-k)/2 + k = (s+k)/2. x is also a positive integer.

    # Iterate through possible values of k. k must be a divisor of N.
    # The condition 4M - k^2 >= 0 implies k <= cuberoot(4N).
    # For N <= 10^18, 4N <= 4*10^18. k <= (4*10^18)^(1/3) approx 1.587 * 10^6.
    # We iterate k up to 2 * 10^6 as a safe upper bound.
    limit_k = 2 * 10**6 

    for k in range(1, limit_k + 1):
        
        # k must be a divisor of N.
        if N % k == 0:
            M = N // k
            
            # Calculate val = 4M - k^2.
            val = 4*M - k*k
            
            # Check the derived conditions:
            # 1. val must be non-negative (required for real solutions for y).
            if val >= 0:
                 # 2. val must be divisible by 3.
                if val % 3 == 0:
                    S = val // 3
                    
                    # 3. S must be a perfect square.
                    is_sq, s = is_perfect_square(S)
                    
                    if is_sq:
                         # 4. s and k must have the same parity for x, y to be integers.
                        if (s % 2) == (k % 2):
                            
                            # 5. s > k is required for y > 0.
                            if s > k:
                                # All conditions met. We found a valid pair (x, y).
                                x = (s + k) // 2
                                y = (s - k) // 2
                                
                                # Since x and y are derived correctly, they satisfy x^3 - y^3 = N.
                                # y = (s-k)/2 >= (k+2-k)/2 = 1 (since s >= k+2 if s>k and same parity)
                                # x = (s+k)/2 > k >= 1. Both x, y are positive integers.
                                
                                print(f"{x} {y}")
                                found = True
                                break # Exit after finding one solution as required.
    
    # If the loop finishes without finding any solution pair.
    if not found:
        print("-1")

# Execute the solve function
solve()

# END YOUR CODE HERE