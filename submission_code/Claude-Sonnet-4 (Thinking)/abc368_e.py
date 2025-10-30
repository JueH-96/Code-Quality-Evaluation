from collections import defaultdict, deque

n, m, x1 = map(int, input().split())
trains = []
for i in range(m):
    a, b, s, t = map(int, input().split())
    trains.append((a, b, s, t))

# Build the constraint graph: graph[i] = [(j, weight), ...] means X_j >= X_i + weight
graph = defaultdict(list)

for i in range(m):
    for j in range(m):
        if i != j:
            a_i, b_i, s_i, t_i = trains[i]
            a_j, b_j, s_j, t_j = trains[j]
            
            # Check if transfer is possible: B_i = A_j and T_i <= S_j
            if b_i == a_j and t_i <= s_j:
                # Constraint: X_j >= X_i + (T_i - S_j)
                weight = t_i - s_j
                graph[i].append((j, weight))

# Initialize X values
x = [0] * m
x[0] = x1

# Use a queue-based approach to propagate updates
queue = deque([0])  # Start with train 0 since its value is fixed
in_queue = [False] * m
in_queue[0] = True

while queue:
    u = queue.popleft()
    in_queue[u] = False
    
    # Propagate updates to all neighbors
    for v, weight in graph[u]:
        new_val = max(x[v], x[u] + weight)
        if new_val > x[v]:
            x[v] = new_val
            if not in_queue[v]:
                queue.append(v)
                in_queue[v] = True

# Output X_2, ..., X_M
result = x[1:]
print(' '.join(map(str, result)))