# YOUR CODE HERE
# Read Team Takahashi's scores for 9 innings
A = list(map(int, input().split()))

# Read Team Aoki's scores for 8 innings
B = list(map(int, input().split()))

# Calculate total scores
takahashi_score = sum(A)
aoki_score_8 = sum(B)

# For Aoki to win: aoki_score_8 + runs_in_9th > takahashi_score
# So minimum runs_in_9th = takahashi_score - aoki_score_8 + 1
min_runs = takahashi_score - aoki_score_8 + 1

print(min_runs)