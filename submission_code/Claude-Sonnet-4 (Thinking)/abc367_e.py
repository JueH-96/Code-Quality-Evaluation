N, K = map(int, input().split())
X = list(map(int, input().split()))
A = list(map(int, input().split()))

# Convert to 0-indexed
X = [x - 1 for x in X]

result = [0] * N

for i in range(N):
    current = i
    seen = {}
    path = []
    
    step = 0
    while current not in seen:
        seen[current] = step
        path.append(current)
        current = X[current]
        step += 1
    
    cycle_start = seen[current]
    cycle_length = step - cycle_start
    
    if K < cycle_start:
        target_pos = path[K]
    else:
        remaining_steps = (K - cycle_start) % cycle_length
        target_pos = path[cycle_start + remaining_steps]
    
    result[i] = A[target_pos]

print(' '.join(map(str, result)))