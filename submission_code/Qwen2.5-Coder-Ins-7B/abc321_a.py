# YOUR CODE HERE
import sys

# Read input from stdin
N = int(sys.stdin.read().strip())

# Convert the number to a string to easily access individual digits
N_str = str(N)

# Check if the digits are strictly decreasing
is_321_like = all(N_str[i] > N_str[i + 1] for i in range(len(N_str) - 1))

# Print the result
print("Yes" if is_321_like else "No")