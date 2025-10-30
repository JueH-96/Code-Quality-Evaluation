# Read the input from stdin
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Calculate the total runs scored by Team Takahashi
total_runs_takahashi = sum(A)

# Calculate the total runs scored by Team Aoki
total_runs_aoki = sum(B)

# Determine the minimum number of runs Team Aoki needs to score to win
# They need to score one more than Team Takahashi's total score
min_runs_to_win = total_runs_takahashi - total_runs_aoki + 1

# Print the result
print(min_runs_to_win)