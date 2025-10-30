from collections import defaultdict

# Read input
N, M = map(int, input().split())
parents = [0] + list(map(int, input().split()))
insurance_data = [list(map(int, input().split())) for _ in range(M)]

# Build parent-child relationships
children = defaultdict(list)
for i in range(2, N+1):
    children[parents[i]].append(i)

# Process insurance data
covered = set()
for x, y in insurance_data:
    queue = [x]
    for _ in range(y):
        new_queue = []
        for person in queue:
            covered.add(person)
            new_queue.extend(children[person])
        queue = new_queue

# Print the answer
print(len(covered))