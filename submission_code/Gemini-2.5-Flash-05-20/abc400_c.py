import sys
import math

# Read the input N from standard input
N = int(sys.stdin.readline())

# Initialize the count of good integers
count = 0

# Start with the smallest possible power of 2 for 'a' (k=1)
# X = 2^k * S^2, where S is an odd positive integer and k >= 1.
# power_of_2 represents 2^k.
power_of_2 = 2  # This corresponds to k=1

# Loop while the current power of 2 is within the limit N
while power_of_2 <= N:
    # We are looking for X = power_of_2 * S^2 such that X <= N.
    # This implies S^2 <= N / power_of_2.
    
    # Calculate the upper bound for S^2.
    # Using integer division ensures we get the floor of N / power_of_2.
    limit_for_S_squared = N // power_of_2
    
    # Calculate the maximum possible integer value for S.
    # S must satisfy S^2 <= limit_for_S_squared.
    # math.sqrt returns a float, and int() truncates it to the integer part,
    # which is equivalent to floor(sqrt(value)). This is correct for max_S.
    max_S = int(math.sqrt(limit_for_S_squared))
    
    # We need to count the number of positive odd integers S
    # such that 1 <= S <= max_S.
    # The count of odd integers in the range [1, M] is (M + 1) // 2.
    # For example:
    # M=1: (1+1)//2 = 1 (S=1)
    # M=2: (2+1)//2 = 1 (S=1)
    # M=3: (3+1)//2 = 2 (S=1, 3)
    # This formula correctly handles cases where max_S is 0 (resulting in 0 count).
    count += (max_S + 1) // 2
    
    # Prepare for the next iteration: move to the next power of 2 (i.e., increment k).
    # Since N can be very large, power_of_2 might grow large, but Python handles
    # arbitrary-precision integers, so overflow is not a concern for power_of_2.
    # The loop condition `power_of_2 <= N` will naturally terminate the loop
    # when power_of_2 exceeds N.
    power_of_2 *= 2

# Print the final count to standard output
sys.stdout.write(str(count) + '
')