# Reading input
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Calculate total runs scored by Team Takahashi and Team Aoki
total_runs_takahashi = sum(A)
total_runs_aoki = sum(B)

# Calculate the minimum runs needed for Team Aoki to win
# Team Aoki needs to have one more run than Team Takahashi to win
runs_needed = total_runs_takahashi - total_runs_aoki + 1

# Output the result
print(runs_needed)