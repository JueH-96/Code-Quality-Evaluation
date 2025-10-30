import sys

# Read the input scores for Team Takahashi
a_scores = list(map(int, sys.stdin.readline().split()))

# Read the input scores for Team Aoki (up to inning 8)
b_scores = list(map(int, sys.stdin.readline().split()))

# Calculate the total score for Team Takahashi at the end of the top of the 9th
takahashi_total_score = sum(a_scores)

# Calculate the total score for Team Aoki at the end of the bottom of the 8th
aoki_current_score = sum(b_scores)

# For Team Aoki to win, their final score must be strictly greater than Team Takahashi's score.
# Let R_9 be the runs scored by Aoki in the bottom of the 9th.
# Aoki's final score = aoki_current_score + R_9
# Condition for Aoki to win: aoki_current_score + R_9 > takahashi_total_score
# We need to find the minimum integer R_9 satisfying this condition.
# R_9 > takahashi_total_score - aoki_current_score
# The minimum integer value for R_9 is (takahashi_total_score - aoki_current_score) + 1

# The problem statement guarantees that takahashi_total_score >= aoki_current_score.
# So, takahashi_total_score - aoki_current_score >= 0.
# Therefore, the minimum runs needed will always be at least 1.

minimum_runs_to_win = takahashi_total_score - aoki_current_score + 1

# Print the result
print(minimum_runs_to_win)