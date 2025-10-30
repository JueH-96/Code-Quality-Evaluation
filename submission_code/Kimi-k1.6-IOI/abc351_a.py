# Read the input values
a = list(map(int, input().split()))
b = list(map(int, input().split()))

# Calculate the total runs for both teams
sum_a = sum(a)
sum_b = sum(b)

# Compute the minimum required runs for Team Aoki to win
required_runs = sum_a - sum_b + 1

# Output the result
print(required_runs)