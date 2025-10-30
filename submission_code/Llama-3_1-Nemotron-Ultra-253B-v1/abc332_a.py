# Read input
n, s, k = map(int, input().split())
total = 0

for _ in range(n):
    p, q = map(int, input().split())
    total += p * q

# Calculate shipping fee
shipping = 0 if total >= s else k

# Output the result
print(total + shipping)