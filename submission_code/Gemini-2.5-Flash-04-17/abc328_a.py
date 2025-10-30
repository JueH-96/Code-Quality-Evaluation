import sys

# Read N and X from the first line
line1 = sys.stdin.readline().split()
N = int(line1[0])
X = int(line1[1])

# Read the N scores from the second line
line2 = sys.stdin.readline().split()
scores = list(map(int, line2))

# Calculate the total score for problems with score <= X
total_score = 0
for score in scores:
    if score <= X:
        total_score += score

# Print the result
print(total_score)