import sys

# Read Team Takahashi's runs for 9 innings
# The first line contains 9 integers A_1, A_2, ..., A_9
a_runs_str = sys.stdin.readline().split()
a_runs = list(map(int, a_runs_str))

# Read Team Aoki's runs for 8 innings
# The second line contains 8 integers B_1, B_2, ..., B_8
b_runs_str = sys.stdin.readline().split()
b_runs = list(map(int, b_runs_str))

# Calculate Team Takahashi's total score after the top of the 9th inning
takahashi_score = sum(a_runs)

# Calculate Team Aoki's total score after the bottom of the 8th inning
aoki_score_before_9th = sum(b_runs)

# Team Aoki needs to score 'x' runs in the bottom of the 9th inning to win.
# Team Aoki's final score will be aoki_score_before_9th + x.
# For Team Aoki to win, their final score must be strictly greater than Team Takahashi's final score.
# So, we need: aoki_score_before_9th + x > takahashi_score
# This inequality can be rewritten as: x > takahashi_score - aoki_score_before_9th

# We are looking for the minimum integer value of x that satisfies this condition.
# The minimum integer x strictly greater than (takahashi_score - aoki_score_before_9th) is
# (takahashi_score - aoki_score_before_9th) + 1.

required_runs = takahashi_score - aoki_score_before_9th + 1

# Print the minimum number of runs Team Aoki needs
print(required_runs)