import sys

def solve(N, K):
    # Initialize the result
    result = []

    # Initialize the current number and the count
    current_number = 1
    current_count = 1

    # Iterate over the range from 1 to N*K
    for _ in range(N*K):
        # Add the current number to the result
        result.append(current_number)

        # If the count is equal to K, reset it and increment the current number
        if current_count == K:
            current_count = 1
            current_number += 1
        else:
            current_count += 1

    return result

# Read the inputs from stdin
N, K = map(int, sys.stdin.readline().split())

# Solve the problem
result = solve(N, K)

# Print the result
print(' '.join(map(str, result)))