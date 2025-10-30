# Read input
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Calculate totals
sum_takahashi = sum(A)
sum_aoki = sum(B)

# Calculate required runs
required = sum_takahashi - sum_aoki + 1

# Output the result
print(required)