import sys

# Read N and M from standard input
N, M = map(int, sys.stdin.readline().split())

# Define the limit for the sum
LIMIT = 10**9

# Initialize the total sum
total_sum = 0

# Flag to track if the sum has exceeded the limit
sum_exceeded = False

# Iterate from i = 0 to M (inclusive)
for i in range(M + 1):
    # Calculate N^i. Python's arbitrary precision integers handle large numbers automatically.
    current_term = N**i

    # Check if adding the 'current_term' to 'total_sum' would exceed the 'LIMIT'.
    # This check is robust:
    # If 'current_term' is very large (e.g., already greater than LIMIT),
    # then 'LIMIT - current_term' will be negative or a very small positive number.
    # Since 'total_sum' is non-negative, the condition 'total_sum > (LIMIT - current_term)'
    # will correctly evaluate to True, indicating that the sum will definitely exceed the LIMIT.
    # If 'current_term' is smaller, this condition correctly checks if 'total_sum' is already
    # close enough to 'LIMIT' that adding 'current_term' would push it over.
    if total_sum > LIMIT - current_term:
        sum_exceeded = True
        break  # The sum will exceed the limit, no need to calculate further

    # Add the current term to the total sum
    total_sum += current_term

# Print the result based on whether the sum exceeded the limit
if sum_exceeded:
    print("inf")
else:
    print(total_sum)