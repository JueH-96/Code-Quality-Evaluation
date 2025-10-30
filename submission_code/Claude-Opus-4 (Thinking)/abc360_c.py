# YOUR CODE HERE
n = int(input())
A = list(map(int, input().split()))
W = list(map(int, input().split()))

# Group items by box
items_in_box = [[] for _ in range(n + 1)]
for i in range(n):
    items_in_box[A[i]].append(W[i])

total_cost = 0
for box in range(1, n + 1):
    if len(items_in_box[box]) > 1:
        # Sort by weight in descending order
        items_in_box[box].sort(reverse=True)
        # Keep the heaviest item, move the rest
        total_cost += sum(items_in_box[box][1:])

print(total_cost)