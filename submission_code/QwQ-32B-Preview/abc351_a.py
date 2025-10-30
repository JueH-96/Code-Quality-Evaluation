# Read first line and split into list of integers
A = list(map(int, input().split()))

# Read second line and split into list of integers
B = list(map(int, input().split()))

# Calculate sums
sum_A = sum(A)
sum_B = sum(B)

# Calculate x
x = sum_A - sum_B + 1

# Print x
print(x)