# YOUR CODE HERE
# Read Team Takahashi's scores (9 innings)
takahashi_scores = list(map(int, input().split()))

# Read Team Aoki's scores (8 innings)
aoki_scores = list(map(int, input().split()))

# Calculate total scores
takahashi_total = sum(takahashi_scores)
aoki_current = sum(aoki_scores)

# Team Aoki needs to score more than Team Takahashi to win
# So they need at least takahashi_total + 1 runs in total
runs_needed = takahashi_total + 1 - aoki_current

# The minimum runs needed in the 9th inning
print(runs_needed)