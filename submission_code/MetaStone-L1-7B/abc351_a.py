# Read the input
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Calculate the total runs for Takahashi and Aoki
T = sum(A)
A_sum = sum(B)

# Compute the minimum runs needed for Aoki
x = (T - A_sum) + 1

# Output the result
print(x)