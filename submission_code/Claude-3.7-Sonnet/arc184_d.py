from collections import deque

n = int(input())
balls = []
for _ in range(n):
    x, y = map(int, input().split())
    balls.append((x, y))

# Precompute comparability
comparable = [[False] * n for _ in range(n)]
for i in range(n):
    for j in range(n):
        if i != j:
            x_i, y_i = balls[i]
            x_j, y_j = balls[j]
            comparable[i][j] = (x_i < x_j and y_i < y_j) or (x_i > x_j and y_i > y_j)

# Simulate all possible sequences of operations using BFS
MOD = 998244353
seen_sets = set()
queue = deque([frozenset(range(n))])  # Start with all balls

while queue:
    curr_set = queue.popleft()
    if curr_set in seen_sets:
        continue
    seen_sets.add(curr_set)
    
    for k in curr_set:
        to_remove = set()
        for i in curr_set:
            if i != k and comparable[i][k]:
                to_remove.add(i)
        
        if to_remove:  # Only compute next_set if there are balls to remove
            next_set = frozenset(curr_set - to_remove)
            if next_set not in seen_sets:
                queue.append(next_set)

answer = len(seen_sets)
print(answer % MOD)