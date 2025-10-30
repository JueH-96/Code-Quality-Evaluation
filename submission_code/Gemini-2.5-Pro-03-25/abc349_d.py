# YOUR CODE HERE
import sys

# Function to find the exponent of the largest power of 2 that divides x
# For x=0, this is effectively infinite. We can represent this with a large number like 60.
# For x>0, calculates k such that 2^k is the largest power of 2 dividing x.
# This is equivalent to finding the number of trailing zeros in binary representation.
def get_max_exponent_dividing(x):
    if x == 0:
        # Based on problem constraints R <= 2^60, we can use 60 as a sufficiently large exponent.
        return 60 
    else:
         # For positive x, x & -x gives the largest power of 2 dividing x, say p=2^k.
         # p.bit_length() - 1 gives k.
         p_div = x & -x 
         k_div = p_div.bit_length() - 1 
         return k_div

def solve():
    # Read input L and R
    L, R = map(int, sys.stdin.readline().split())

    # List to store the resulting intervals (pairs)
    ans = []
    # Current position, starts at L
    curr = L
    
    # Continue as long as the current position is less than R
    while curr < R:
        
        # Calculate the exponent k_div for the largest power of 2 that divides curr
        k_div = get_max_exponent_dividing(curr)

        # Start with k = k_div. This represents the largest possible step size 2^k
        # based purely on the divisibility property of curr.
        k = k_div
             
        # Now we need to ensure that taking a step of size 2^k does not overshoot R.
        # The interval is [curr, curr + 2^k). The endpoint must be <= R.
        # So, curr + 2^k <= R must hold.
        
        # If curr + 2^k > R, the step size 2^k is too large.
        # We need to decrease k (use smaller powers of 2) until the condition curr + 2^k <= R is met.
        # This loop finds the largest k such that k <= k_div AND curr + 2^k <= R.
        while curr + (1 << k) > R:
            k -= 1
            # It's guaranteed that k will eventually become small enough (e.g., k=0, step=1)
            # unless R = curr, but the loop condition `curr < R` prevents that starting state.
            # If k becomes negative, 1 << k would result in 0. This case should technically not be reached
            # if R > curr because 2^0 = 1, and curr + 1 <= R must hold if R > curr.
            if k < 0:
                 # Safety break, though unlikely given L < R.
                 break
        
        # Check if k became negative. If so, something is wrong. Exit.
        if k < 0:
            break

        # The correct exponent k is found. Calculate the step size 2^k.
        p_step = 1 << k
             
        # Safety check: Ensure step is positive to prevent infinite loop.
        # If R > curr, the remaining distance is at least 1. The largest power of 2 <= R - curr
        # will be at least 2^0 = 1. So p_step should always be >= 1.
        if p_step == 0: 
            # If p_step is 0, it implies k < 0, which indicates an issue.
            break

        # Add the interval [curr, curr + p_step) to the answer list
        ans.append((curr, curr + p_step))
        
        # Move the current position forward by the step size
        curr += p_step

    # Print the number of intervals found
    print(len(ans))
    # Print each interval pair (l, r)
    for l, r in ans:
        print(f"{l} {r}")

# Execute the solve function
solve()