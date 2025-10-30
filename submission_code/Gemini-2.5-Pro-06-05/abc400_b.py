# YOUR CODE HERE
import sys

# Read the two positive integer inputs N and M from standard input.
# sys.stdin.readline() is often used in competitive programming for fast I/O.
try:
    N, M = map(int, sys.stdin.readline().split())
except (IOError, ValueError):
    # According to the problem specification, the input will always be valid.
    # This error handling is for general robustness and would not be strictly
    # necessary in a contest environment with guaranteed-valid inputs.
    sys.exit(1)


# The problem defines a threshold value of 10^9.
LIMIT = 10**9

# The value to calculate is the sum of a geometric series: X = N^0 + N^1 + ... + N^M.

# First, handle the special case where the common ratio N is 1.
if N == 1:
    # The sum simplifies to 1 + 1 + ... + 1, repeated M+1 times.
    result = M + 1
    # Given the constraint M <= 100, the result (M+1) will be at most 101.
    # This is always less than or equal to LIMIT, so we can print it directly.
    print(result)
else:
    # For N > 1, the terms of the series grow. We compute the sum
    # iteratively and check if it exceeds the LIMIT at each step. This avoids
    # calculating potentially huge numbers if the sum overflows early.
    
    # Initialize the sum with the first term of the series, N^0 = 1.
    total_sum = 1
    # This variable will hold the current term, N^i. It also starts at N^0 = 1.
    current_power = 1
    
    # The for-else structure is used to handle the two possible outcomes:
    # 1. The loop breaks because the sum exceeds the limit ("inf" case).
    # 2. The loop completes, and the final sum is within the limit.
    
    # We loop M times to add the terms from N^1 to N^M.
    for _ in range(M):
        # Optimization: Before calculating the next power, we check if it
        # would be excessively large. If current_power * N > LIMIT, the total
        # sum will also exceed LIMIT. We rewrite this as `current_power > LIMIT // N`
        # to use safe integer division. This provides an efficient early exit.
        if current_power > LIMIT // N:
            print("inf")
            break
        
        # Calculate the next term: N^(i+1) = N^i * N.
        current_power *= N
        
        # Add the new term to the running total.
        total_sum += current_power
        
        # After adding the term, check if the cumulative sum has exceeded the limit.
        if total_sum > LIMIT:
            print("inf")
            break
    else:
        # The 'else' block of a for-loop executes only if the loop
        # was not terminated by a 'break' statement.
        # This implies the sum never exceeded the LIMIT.
        print(total_sum)