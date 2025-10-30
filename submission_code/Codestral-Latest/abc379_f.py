import sys
from collections import deque

input = sys.stdin.read
data = input().split()

index = 0
N = int(data[index])
Q = int(data[index + 1])
index += 2

H = [int(data[index + i]) for i in range(N)]
index += N

queries = []
for _ in range(Q):
    l = int(data[index])
    r = int(data[index + 1])
    queries.append((l, r))
    index += 2

# Precompute the next greater element to the right for each building
next_greater = [-1] * N
stack = deque()
for i in range(N - 1, -1, -1):
    while stack and H[stack[0]] < H[i]:
        stack.popleft()
    if stack:
        next_greater[i] = stack[0]
    stack.appendleft(i)

# Precompute the next greater element to the left for each building
prev_greater = [-1] * N
stack = deque()
for i in range(N):
    while stack and H[stack[0]] < H[i]:
        stack.popleft()
    if stack:
        prev_greater[i] = stack[0]
    stack.appendleft(i)

# Function to count visible buildings to the east of r from both l and r
def count_visible_buildings(l, r):
    count = 0
    current = r
    while current < N:
        if next_greater[current] == -1 or next_greater[current] > r:
            count += 1
        current = next_greater[current]
    return count

# Process each query
results = []
for l, r in queries:
    results.append(count_visible_buildings(l, r))

# Output the results
for result in results:
    print(result)