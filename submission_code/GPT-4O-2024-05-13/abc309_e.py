# YOUR CODE HERE
import sys
from collections import defaultdict, deque

input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])

parents = [0] * (N + 1)
for i in range(2, N + 1):
    parents[i] = int(data[i - 1])

insurances = []
for i in range(M):
    x = int(data[N + 2 * i])
    y = int(data[N + 2 * i + 1])
    insurances.append((x, y))

# Build the tree
children = defaultdict(list)
for i in range(2, N + 1):
    children[parents[i]].append(i)

# BFS to find all descendants within y generations
def find_covered(x, y):
    covered = set()
    queue = deque([(x, 0)])
    while queue:
        current, generation = queue.popleft()
        if generation > y:
            continue
        covered.add(current)
        for child in children[current]:
            queue.append((child, generation + 1))
    return covered

# Find all covered people
covered_people = set()
for x, y in insurances:
    covered_people.update(find_covered(x, y))

print(len(covered_people))