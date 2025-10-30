import sys

# Read input
N, M = map(int, input().split())
P = list(map(int, input().split()))

# Define a function to calculate the total cost for buying k units of the i-th product
def cost(i, k):
    return k ** 2 * P[i]

# Initialize the maximum number of units that can be bought
max_units = 0

# Iterate through each product
for i in range(N):
    # Binary search to find the maximum number of units of the i-th product that can be bought
    left, right = 0, 10 ** 100
    while left <= right:
        mid = (left + right) // 2
        total_cost = sum(cost(j, mid) for j in range(i))
        if total_cost + cost(i, mid) <= M:
            max_units += mid
            left = mid + 1
        else:
            right = mid - 1

# Print the answer
print(max_units)