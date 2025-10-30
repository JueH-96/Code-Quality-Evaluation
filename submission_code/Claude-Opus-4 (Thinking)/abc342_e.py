N, M = map(int, input().split())

# Store train information
trains = []
for _ in range(M):
    l, d, k, c, A, B = map(int, input().split())
    trains.append((l, d, k, c, A, B))

# f[i] = latest departure time from station i to reach station N
f = [-float('inf')] * (N + 1)
f[N] = float('inf')

# Iterate until convergence
changed = True
while changed:
    changed = False
    for l, d, k, c, A, B in trains:
        if f[B] == -float('inf'):
            continue
        
        # Find the latest train from A to B that arrives by time f[B]
        if f[B] == float('inf'):
            # Can take the latest train
            i = k - 1
        else:
            # We want the largest i such that l + i*d + c <= f[B]
            # So i <= (f[B] - l - c) / d
            max_arrival = f[B]
            if max_arrival < l + c:
                # Even the first train arrives too late
                continue
            
            # Calculate the index of the latest valid train
            i = min((max_arrival - l - c) // d, k - 1)
        
        dep_time = l + i * d
        
        if dep_time > f[A]:
            f[A] = dep_time
            changed = True

# Output results
for i in range(1, N):
    if f[i] == -float('inf'):
        print("Unreachable")
    else:
        print(f[i])