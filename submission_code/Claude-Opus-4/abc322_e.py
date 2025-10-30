# YOUR CODE HERE
N, K, P = map(int, input().split())

plans = []
for i in range(N):
    line = list(map(int, input().split()))
    cost = line[0]
    values = line[1:]
    plans.append((cost, values))

min_cost = float('inf')
found = False

# Try all possible subsets of development plans
for mask in range(1 << N):
    # Calculate total values for each parameter
    param_values = [0] * K
    total_cost = 0
    
    for i in range(N):
        if mask & (1 << i):
            total_cost += plans[i][0]
            for j in range(K):
                param_values[j] += plans[i][1][j]
    
    # Check if all parameters meet the requirement
    valid = True
    for j in range(K):
        if param_values[j] < P:
            valid = False
            break
    
    if valid:
        found = True
        min_cost = min(min_cost, total_cost)

if found:
    print(min_cost)
else:
    print(-1)