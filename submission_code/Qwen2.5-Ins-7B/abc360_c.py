# YOUR CODE HERE
from collections import Counter

N = int(input())
boxes = list(map(int, input().split()))
weights = list(map(int, input().split()))

box_counts = Counter(boxes)
total_cost = 0

for i in range(1, N + 1):
    min_cost = float('inf')
    for j in range(1, N + 1):
        if box_counts[j] > 0:
            min_cost = min(min_cost, weights[i - 1])
            box_counts[j] -= 1
    total_cost += min_cost

print(total_cost)