import sys  # Not needed, but can be included if using sys.stdin; however, using built-in input is sufficient.

# Read the first line and parse N and X
first_line = input().split()
N = int(first_line[0])
X = int(first_line[1])

# Read the second line and parse the scores
second_line = input().split()
scores = [int(s) for s in second_line]

# Calculate the sum of scores that are less than or equal to X
total_sum = sum(score for score in scores if score <= X)

# Print the total sum
print(total_sum)