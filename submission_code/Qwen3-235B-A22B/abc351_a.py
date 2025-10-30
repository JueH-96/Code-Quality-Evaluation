# Read the input line and split into integers
scores = list(map(int, input().split()))

# Calculate Team Takahashi's total runs (sum of first 9 values)
sum_takahashi = sum(scores[:9])

# Calculate Team Aoki's runs after 8 innings (sum of next 8 values)
sum_aoki = sum(scores[9:])

# Calculate the minimum required runs for Team Aoki to win
required_runs = sum_takahashi - sum_aoki + 1

# Print the result
print(required_runs)