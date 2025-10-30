# YOUR CODE HERE
N = int(input().strip())

# Find all divisors of N between 1 and 9
divisors = [j for j in range(1, 10) if N % j == 0]

# Initialize the result string
result = []

# Iterate over each i from 0 to N
for i in range(N + 1):
    # Find the smallest divisor j such that i is a multiple of N/j
    smallest_divisor = None
    for j in divisors:
        if i % (N // j) == 0:
            if smallest_divisor is None or j < smallest_divisor:
                smallest_divisor = j

    # Append the corresponding character to the result string
    if smallest_divisor is not None:
        result.append(str(smallest_divisor))
    else:
        result.append('-')

# Print the result string
print(''.join(result))