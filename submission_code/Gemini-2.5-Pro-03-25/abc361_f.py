# YOUR CODE HERE
import sys
import math

# Function to compute integer cube root using binary search
def integer_cube_root(N):
    """
    Computes the largest integer k such that k^3 <= N.
    Uses binary search for efficiency.
    """
    # Handles N < 1 based on typical integer root definitions. 
    # For this problem constraints N >= 1, so this check covers N=0 if it were possible.
    if N < 1: return 0 
    
    # Set binary search range. The maximum possible cube root for N <= 10^18 is 10^6.
    # Using 10^6 + 3 as the high bound ensures it covers 10^6 and provides some buffer.
    # Any value slightly larger than 10^6 would work as an initial upper bound.
    low = 1
    high = 10**6 + 3 
    
    best_k = 0 # Stores the largest k found so far such that k^3 <= N
    while low <= high:
        mid = (low + high) // 2
        # Avoid mid=0 edge case for safety, although low starts at 1.
        if mid == 0: break 

        # Calculate mid^3 safely using Python's arbitrary precision integers.
        # Python automatically handles large numbers, so standard multiplication is fine.
        
        # Optimization for small N: if mid itself is already larger than N, then mid^3 > N.
        # This avoids potentially large calculations when mid is clearly too big.
        if mid > N : # Since N >= 1, mid > N implies mid >= 2.
             # If mid > N, then mid*mid*mid > N*N*N >= N (since N >= 1).
             # So we can treat val as if it exceeds N.
             val = N + 1 
        else:
             # Calculate the cube value. This is safe up to Python's memory limits.
             val = mid * mid * mid

        if val <= N:
            # If mid^3 <= N, then 'mid' is a possible candidate for the integer cube root.
            # We store it as the best candidate found so far and try searching for potentially larger valid k values.
            best_k = mid  
            low = mid + 1 # Search in the upper half [mid+1, high]
        else: # val > N
            # If mid^3 > N, then 'mid' is too large. We need to search for smaller k values.
            high = mid - 1 # Search in the lower half [low, mid-1]
            
    # After the binary search loop terminates, best_k holds the largest integer k such that k^3 <= N.
    return best_k


def solve():
    # Read the input integer N from standard input
    N = int(sys.stdin.readline())

    # Calculate floor(sqrt(N)) using math.isqrt for efficiency and accuracy with large integers.
    # This gives the count of perfect squares from 1^2 up to k^2 <= N.
    # Let S2 be the set of perfect squares <= N. Then |S2| = sqrt_N. This includes 1 = 1^2.
    sqrt_N = math.isqrt(N)
    
    # Calculate the maximum possible base 'a' such that a^3 <= N.
    # This is needed to limit the range of the base 'a' when generating powers a^b with b >= 3.
    max_a_for_cube = integer_cube_root(N)

    # Initialize a set to store distinct perfect powers x = a^b where a >= 2, b >= 3, and x <= N.
    # This set represents S'_ge3 = { x | x = a^b, a >= 2, b >= 3, x <= N }.
    # Using a set automatically handles duplicate values (e.g., 64 = 4^3 = 2^6).
    set_ge3 = set()
    
    # Iterate through possible bases 'a' starting from 2 up to the maximum base determined for cubes.
    # If a > max_a_for_cube, then a^3 > N, so a^b > N for all b >= 3.
    for a in range(2, max_a_for_cube + 1):
        
        # Start calculating powers from a^3.
        val = a * a * a 
        
        # Inner loop computes higher powers a^b for b = 3, 4, 5, ...
        # This loop continues as long as the calculated power `val` does not exceed N.
        while True:
            # Check if the current power `val` is greater than N. If so, stop for this base 'a'.
            if val > N: 
                 break

            # Add the valid perfect power `val` to our set.
            set_ge3.add(val)
            
            # Prepare to calculate the next power: val * a (which would be a^(current_exponent + 1))
            # Check BEFORE performing the multiplication to see if the result would exceed N.
            # This check uses integer division `N // val` for safety with large integers.
            # If a > N / val (mathematically equivalent to a * val > N for positive integers), 
            # then the next power will exceed N.
            
            # Safety check for val == 0, although this should not happen for a >= 2.
            if val == 0:
                break 
            
            # The core check to prevent unnecessary computation and potential overflow if not using large integers.
            if a > N // val: 
                 break # Stop computing powers for this base 'a' as the next one will be too large.
            
            # Calculate the next power value if it's potentially within range.
            val *= a

    # Calculate |S'_ge3|, which is the number of distinct perfect powers found 
    # with base a >= 2 and exponent b >= 3.
    size_S_ge3 = len(set_ge3)

    # Calculate the size of the intersection: |S2 intersect S'_ge3|.
    # This counts how many elements in the set `set_ge3` are also perfect squares.
    count_intersection = 0
    for x in set_ge3:
        # Check if x is a perfect square using integer square root `math.isqrt`.
        k = math.isqrt(x)
        # If k*k == x, then x is a perfect square. Increment the counter.
        if k * k == x:
            count_intersection += 1

    # Final Answer Calculation:
    # The total count of integers x (1 <= x <= N) expressible as a^b (a>=1, b>=2)
    # is determined using the Principle of Inclusion-Exclusion.
    # Let S be the set of all such integers. S = S2 U S_ge3, where S_ge3 is { a^b | a>=1, b>=3 }.
    # We calculated |S2| = sqrt_N (includes 1) and S'_ge3 = { a^b | a>=2, b>=3 }.
    # The total count is |S| = |S2| + |S'_ge3| - |S2 intersect S'_ge3|.
    # This counts all squares (via sqrt_N), adds distinct higher powers (size_S_ge3),
    # and subtracts those higher powers that were already counted as squares (count_intersection).
    # This formula correctly handles the inclusion of 1 and avoids double counting.
    total_count = sqrt_N + size_S_ge3 - count_intersection
    
    # Print the final computed count to standard output.
    print(total_count)

# Execute the main solver function when the script runs.
solve()