def solve():
    n = int(input())
    constraints = []
    for _ in range(n):
        a, b, c = map(int, input().split())
        constraints.append((a, b, c))
    
    # Check if there's any solution at all
    for a, b, c in constraints:
        if a + b >= c:
            return 0
    
    # Find the upper bound for x
    max_x = min((c - 1) // a for a, b, c in constraints)
    
    if max_x < 1:
        return 0
    
    total = 0
    for x in range(1, max_x + 1):
        # Find the maximum valid y for this x
        min_max_y = float('inf')
        valid = True
        for a, b, c in constraints:
            remainder = c - a * x
            if remainder <= 0:
                valid = False
                break
            max_y_for_this_constraint = (remainder - 1) // b
            min_max_y = min(min_max_y, max_y_for_this_constraint)
        
        if not valid:
            break  # No more valid x values
        
        if min_max_y >= 1:
            total += min_max_y
        else:
            break  # No more valid (x, y) pairs
    
    return total

t = int(input())
for _ in range(t):
    print(solve())