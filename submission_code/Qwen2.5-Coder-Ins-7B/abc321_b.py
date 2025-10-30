# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
X = int(data[1])
scores = list(map(int, data[2:]))

# Sort the scores to easily find the minimum score needed
scores.sort()

# Calculate the sum of the scores excluding the highest and lowest
current_sum = sum(scores[1:-1])

# Calculate the minimum score needed in the last round
min_score_needed = X - current_sum

# Check if the minimum score needed is valid
if min_score_needed < 0 or min_score_needed > 100:
    print(-1)
else:
    print(min_score_needed)