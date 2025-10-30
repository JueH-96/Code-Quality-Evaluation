# Read the input
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Calculate the sums
sum_takahashi = sum(A)
sum_aoki = sum(B)

# Compute the required runs
required_runs = sum_takahashi - sum_aoki + 1

# Output the result
print(required_runs)