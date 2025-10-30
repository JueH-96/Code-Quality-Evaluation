# YOUR CODE HERE
n = int(input())
boxes = list(map(int, input().split()))
weights = list(map(int, input().split()))

# Create a list to store items in each box
items_in_box = [[] for _ in range(n + 1)]

# Group items by their current box
for i in range(n):
    box = boxes[i]
    weight = weights[i]
    items_in_box[box].append((weight, i))

# Sort items in each box by weight (ascending)
for i in range(1, n + 1):
    items_in_box[i].sort()

total_cost = 0

# For each box with multiple items, move out all but the heaviest
for i in range(1, n + 1):
    if len(items_in_box[i]) > 1:
        # Keep the heaviest item (last after sorting), move others
        for j in range(len(items_in_box[i]) - 1):
            total_cost += items_in_box[i][j][0]

print(total_cost)