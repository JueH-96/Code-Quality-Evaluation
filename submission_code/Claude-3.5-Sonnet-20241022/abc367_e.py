N, K = map(int, input().split())
X = list(map(int, input().split()))
A = list(map(int, input().split()))

# Find cycle length and starting point
seen = {}
curr = A[:]
step = 0

while True:
    # Convert list to tuple for hashing
    curr_tuple = tuple(curr)
    if curr_tuple in seen:
        cycle_start = seen[curr_tuple]
        cycle_len = step - cycle_start
        break
    seen[curr_tuple] = step
    
    # Perform one operation
    next_arr = [0] * N
    for i in range(N):
        next_arr[i] = curr[X[i]-1]
    curr = next_arr
    step += 1
    
    # If K is small, we might finish before finding cycle
    if step == K:
        print(*curr)
        exit()

# Calculate remaining operations after cycle detection
if K < cycle_start:
    # K is before cycle starts
    curr = A[:]
    for _ in range(K):
        next_arr = [0] * N
        for i in range(N):
            next_arr[i] = curr[X[i]-1]
        curr = next_arr
else:
    # Skip to equivalent position in cycle
    remaining = (K - cycle_start) % cycle_len
    curr = A[:]
    for _ in range(cycle_start + remaining):
        next_arr = [0] * N
        for i in range(N):
            next_arr[i] = curr[X[i]-1]
        curr = next_arr

print(*curr)