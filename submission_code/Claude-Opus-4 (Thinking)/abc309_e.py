from collections import defaultdict, deque

# Read input
N, M = map(int, input().split())
parents = list(map(int, input().split()))

# Build the tree
children = defaultdict(list)
for i in range(2, N + 1):
    children[parents[i - 2]].append(i)

# Process insurances
covered = set()
for _ in range(M):
    x, y = map(int, input().split())
    # BFS to find all descendants up to y generations
    queue = deque([(x, 0)])  # (person, generation)
    while queue:
        person, gen = queue.popleft()
        covered.add(person)
        if gen < y:
            for child in children[person]:
                queue.append((child, gen + 1))

# Output the answer
print(len(covered))