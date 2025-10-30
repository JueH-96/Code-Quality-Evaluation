# Read input from stdin
import sys

# Read N, M, and P from a single line of input
N, M, P = map(int, sys.stdin.readline().split())

# Initialize a counter for the number of full moon days
count = 0

# Start checking from the first full moon day, which is M.
current_day = M

# Loop through the sequence of full moon days: M, M+P, M+2P, ...
# Continue as long as the current full moon day is less than or equal to N.
# According to constraints, M >= 1 and P >= 1.
# The sequence of full moon days is M, M+P, M+2P, ..., which means the days are always >= M.
# Since M >= 1, all full moon days are automatically >= 1.
# Thus, we only need to check the upper bound N.
while current_day <= N:
    # If the current day is within the range [1, N] (which is guaranteed for >= M
    # and <= N), it's a valid full moon day to count.
    count += 1

    # Calculate the next full moon day by adding the period P.
    current_day += P

# Print the total number of full moon days found within the range [1, N].
print(count)