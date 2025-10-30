# YOUR CODE HERE
import sys

# Read input
S = list(map(int, sys.stdin.read().strip().split()))

# Check if the sequence is monotonically non-decreasing
is_non_decreasing = all(S[i] <= S[i+1] for i in range(7))

# Check if all elements are between 100 and 675 inclusive
in_range = all(100 <= x <= 675 for x in S)

# Check if all elements are multiples of 25
multiples_of_25 = all(x % 25 == 0 for x in S)

# Print the result based on the conditions
if is_non_decreasing and in_range and multiples_of_25:
    print("Yes")
else:
    print("No")