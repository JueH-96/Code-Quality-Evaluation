# YOUR CODE HERE
import sys

# Read input from stdin
input_line = sys.stdin.read().strip()
# Split the input line into a list of integers
S = list(map(int, input_line.split()))

# Check if the sequence is monotonically non-decreasing
is_non_decreasing = all(S[i] <= S[i + 1] for i in range(len(S) - 1))

# Check if all elements are between 100 and 675, inclusive
are_between_100_and_675 = all(100 <= x <= 675 for x in S)

# Check if all elements are multiples of 25
are_multiples_of_25 = all(x % 25 == 0 for x in S)

# Print Yes if all conditions are satisfied, otherwise print No
if is_non_decreasing and are_between_100_and_675 and are_multiples_of_25:
    print("Yes")
else:
    print("No")