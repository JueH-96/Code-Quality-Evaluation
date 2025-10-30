# Read input
A = list(map(int, input().split()))
B = list(map(int, input().split()))

sum_a = sum(A)
sum_b = sum(B)

# Calculate the minimum runs needed for Aoki to win
required = sum_a - sum_b + 1

print(required)