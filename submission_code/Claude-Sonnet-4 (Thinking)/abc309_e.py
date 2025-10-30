from collections import defaultdict, deque

N, M = map(int, input().split())

# Build parent-child relationships
children = defaultdict(list)
if N > 1:
    parents = list(map(int, input().split()))
    for i, parent in enumerate(parents, 2):
        children[parent].append(i)

# Set to track covered people
covered = set()

# Process each insurance
for _ in range(M):
    x, y = map(int, input().split())
    
    # BFS to find all people covered by this insurance
    queue = deque([(x, 0)])  # (person, generation)
    
    while queue:
        person, generation = queue.popleft()
        covered.add(person)
        
        if generation < y:
            for child in children[person]:
                queue.append((child, generation + 1))

print(len(covered))