import sys
from heapq import heappop, heappush

input = sys.stdin.readline

N = int(input())
A = list(map(int, input().split()))
W = list(map(int, input().split()))

# Adjust indices to be 0-based
A = [a - 1 for a in A]

# Create a list of lists to store the weights of items in each box
boxes = [[] for _ in range(N)]
for i in range(N):
    heappush(boxes[A[i]], -W[i])

# Count the number of boxes with more than one item
excess_boxes = 0
for box in boxes:
    if len(box) > 1:
        excess_boxes += 1

# Create a min-heap to store the weights of items that can be moved
min_heap = []
for box in boxes:
    if len(box) > 0:
        heappush(min_heap, -box[0])

# Calculate the minimum cost to make each box contain exactly one item
cost = 0
while excess_boxes > 0:
    # Move the lightest item from a box with more than one item
    weight = -heappop(min_heap)
    cost += weight
    excess_boxes -= 1

print(cost)