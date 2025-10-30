# YOUR CODE HERE
# Read input values
N, S, K = map(int, input().split())
total = 0
for _ in range(N):
    P, Q = map(int, input().split())
    total += P * Q

# Calculate shipping fee
if total < S:
    total += K

# Output the result
print(total)