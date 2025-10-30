N = int(input())
A = list(map(int, input().split()))
W = list(map(int, input().split()))

# Group items by current box
boxes = [[] for _ in range(N)]
for i in range(N):
    boxes[A[i] - 1].append(W[i])

# For each box with multiple items, move out all but the heaviest
total_cost = 0
for j in range(N):
    if len(boxes[j]) > 1:
        boxes[j].sort()
        # Move out all but the heaviest item
        total_cost += sum(boxes[j][:-1])

print(total_cost)