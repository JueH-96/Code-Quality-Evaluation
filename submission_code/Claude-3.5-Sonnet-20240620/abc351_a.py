# YOUR CODE HERE
# Read Takahashi's scores
takahashi_scores = list(map(int, input().split()))

# Read Aoki's scores
aoki_scores = list(map(int, input().split()))

# Calculate total scores
takahashi_total = sum(takahashi_scores)
aoki_total = sum(aoki_scores)

# Calculate minimum runs needed for Aoki to win
min_runs = takahashi_total - aoki_total + 1

# Print the result
print(min_runs)