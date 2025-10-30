import sys

# Read input
N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Sort the boxes in ascending order of price
boxes = sorted([(a, i+1) for i, a in enumerate(A)])

# Initialize the total cost to 0
total_cost = 0

# Iterate through the persons
for i in range(M):
    # Find the cheapest box that has at least B[i] candies
    for price, box_num in boxes:
        if price >= B[i]:
            total_cost += price
            break
    else:
        # If no such box is found, it's impossible to satisfy the condition
        print(-1)
        return

# Print the minimum total amount of money Takahashi needs to pay
print(total_cost)