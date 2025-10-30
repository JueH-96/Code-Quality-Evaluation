def solve(N):
    # Base case
    if N == 1:
        return 0
        
    # Split N into two parts and add N to cost
    cost = N
    left = N // 2
    right = (N + 1) // 2
    
    # Recursively solve for both parts if they are >= 2
    if left >= 2:
        cost += solve(left)
    if right >= 2:
        cost += solve(right)
        
    return cost

N = int(input())
print(solve(N))