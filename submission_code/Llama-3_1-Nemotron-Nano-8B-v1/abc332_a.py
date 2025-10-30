# Read input values
n, s, k = map(int, input().split())

total = 0
for _ in range(n):
    p, q = map(int, input().split())
    total += p * q

# Determine shipping fee
shipping = 0 if total >= s else k

# Calculate and print the result
print(total + shipping)