# YOUR CODE HERE
n, x = map(int, input().split())
scores = list(map(int, input().split()))

# Sort the scores
scores.sort()

# Calculate the current sum without the highest and lowest scores
current_sum = sum(scores[1:-1])

# Calculate the minimum score needed in the last round
min_score_needed = x - current_sum

# Check if the minimum score needed is valid
if min_score_needed < 0 or min_score_needed > 100:
    print(-1)
else:
    print(max(0, min_score_needed))