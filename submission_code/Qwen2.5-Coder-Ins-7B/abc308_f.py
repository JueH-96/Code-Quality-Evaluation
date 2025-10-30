# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])

P = list(map(int, data[2:2+N]))
L = list(map(int, data[2+N:2+N+M]))
D = list(map(int, data[2+N+M:]))

# Sort items by their regular price in ascending order
items = sorted(range(N), key=lambda i: P[i])

# Sort coupons by their discount in descending order
coupons = sorted(range(M), key=lambda i: D[i], reverse=True)

# Initialize variables
total_cost = 0
used_coupons = set()

# Iterate over items and apply coupons
for i in items:
    if len(used_coupons) == M:
        # No more coupons available, buy at regular price
        total_cost += P[i]
    else:
        # Try to find a suitable coupon for the current item
        for j in range(M):
            if j not in used_coupons and L[j] <= P[i]:
                # Use the coupon and mark it as used
                total_cost += P[i] - D[j]
                used_coupons.add(j)
                break
        else:
            # No suitable coupon found, buy at regular price
            total_cost += P[i]

# Print the total cost
print(total_cost)