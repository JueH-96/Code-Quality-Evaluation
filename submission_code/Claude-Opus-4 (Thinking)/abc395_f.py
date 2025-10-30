# YOUR CODE HERE
def is_feasible(N, X, U, D, H):
    # Check if H is feasible
    # For each i, new U_i must be in [max(0, H - D[i]), min(U[i], H)]
    
    # Start with the valid range for position 0
    low = max(0, H - D[0])
    high = min(U[0], H)
    
    if low > high:
        return False
    
    for i in range(1, N):
        # Valid range for position i based on H
        valid_low = max(0, H - D[i])
        valid_high = min(U[i], H)
        
        if valid_low > valid_high:
            return False
        
        # Reachable range from previous position
        reachable_low = low - X
        reachable_high = high + X
        
        # Intersection
        new_low = max(valid_low, reachable_low)
        new_high = min(valid_high, reachable_high)
        
        if new_low > new_high:
            return False
        
        low = new_low
        high = new_high
    
    return True

N, X = map(int, input().split())
U = []
D = []
for _ in range(N):
    u, d = map(int, input().split())
    U.append(u)
    D.append(d)

# The maximum possible H is min(U[i] + D[i])
max_H = min(U[i] + D[i] for i in range(N))

# Binary search for the maximum feasible H
left, right = 0, max_H
best_H = 0

while left <= right:
    mid = (left + right) // 2
    if is_feasible(N, X, U, D, mid):
        best_H = mid
        left = mid + 1
    else:
        right = mid - 1

# Calculate the cost
total_sum = sum(U[i] + D[i] for i in range(N))
cost = total_sum - N * best_H

print(cost)