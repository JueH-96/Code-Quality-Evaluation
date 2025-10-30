from collections import defaultdict, deque

N, M = map(int, input().split())
parents = list(map(int, input().split()))
children = defaultdict(list)
for i, p in enumerate(parents, start=2):
    children[p].append(i)

insurances = []
for _ in range(M):
    x, y = map(int, input().split())
    insurances.append((x, y))

# build a tree with depth information
depth = {1: 0}
queue = deque([1])
while queue:
    parent = queue.popleft()
    for child in children[parent]:
        depth[child] = depth[parent] + 1
        queue.append(child)

# for each insurance, mark the covered people
covered = set()
for x, y in insurances:
    for i in range(x, N+1):
        if i not in depth or depth[i] > depth[x] + y:
            break
        covered.add(i)

print(len(covered))