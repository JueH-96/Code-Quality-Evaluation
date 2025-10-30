# Read the input for Team Takahashi's runs
a = list(map(int, input().split()))
sum_a = sum(a)

# Read the input for Team Aoki's runs
b = list(map(int, input().split()))
sum_b = sum(b)

# Calculate the minimum required runs
required = sum_a - sum_b + 1

# Output the result
print(required)