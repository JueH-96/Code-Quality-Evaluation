# Read the runs for Team Takahashi and Team Aoki
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Calculate the total runs for each team
sum_A = sum(A)
sum_B = sum(B)

# Compute the minimum runs needed for Team Aoki to win
required_runs = (sum_A - sum_B) + 1

print(required_runs)