# Read the input values
a = list(map(int, input().split()))
b = list(map(int, input().split()))

# Calculate the total runs for each team
sum_a = sum(a)
sum_b = sum(b)

# Determine the minimum runs needed for Team Aoki to win
required_runs = sum_a - sum_b + 1

# Output the result
print(required_runs)