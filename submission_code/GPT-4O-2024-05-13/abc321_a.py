# YOUR CODE HERE
def is_321_like_number(N):
    N_str = str(N)
    for i in range(len(N_str) - 1):
        if N_str[i] <= N_str[i + 1]:
            return "No"
    return "Yes"

# Read input
import sys
input = sys.stdin.read().strip()

# Check if the number is 321-like and print the result
print(is_321_like_number(input))