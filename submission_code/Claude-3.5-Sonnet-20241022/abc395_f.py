def solve():
    # Read input
    N, X = map(int, input().split())
    teeth = []
    for _ in range(N):
        u, d = map(int, input().split())
        teeth.append((u, d))
    
    # For each possible height H, try to make all teeth sum to H
    # while maintaining the difference constraint
    def try_height(H):
        # For each upper tooth value, we need to find what value it should be
        # such that upper[i] + lower[i] = H and |upper[i] - upper[i+1]| ≤ X
        
        # For each position, calculate the range of possible values for upper tooth
        ranges = []
        for i in range(N):
            # Upper tooth can't be negative and can't exceed H
            min_val = 0
            max_val = H
            
            # Upper + lower must equal H
            u, d = teeth[i]
            if d > H:  # Impossible case
                return float('inf')
            # Upper tooth must be exactly H - d
            target = H - d
            if target < 0:
                return float('inf')
            
            ranges.append((target, u))  # (target value, original value)
        
        # Check if adjacent differences can be satisfied
        for i in range(N-1):
            target1, orig1 = ranges[i]
            target2, orig2 = ranges[i+1]
            if abs(target1 - target2) > X:
                return float('inf')
        
        # Calculate total cost
        cost = 0
        for target, orig in ranges:
            if orig > target:
                cost += orig - target
        
        return cost

    # Try all possible heights
    # Height must be between min(u+d) and max(u+d)
    min_h = min(u + d for u, d in teeth)
    max_h = max(u + d for u, d in teeth)
    
    ans = float('inf')
    for h in range(min_h, max_h + 1):
        ans = min(ans, try_height(h))
    
    return ans

print(solve())