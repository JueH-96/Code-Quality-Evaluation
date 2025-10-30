# YOUR CODE HERE
def solve_case():
    n = int(input())
    constraints = []
    for _ in range(n):
        a, b, c = map(int, input().split())
        constraints.append((a, b, c))
    
    # Find max x (when y = 1, we need a*x + b*1 < c, so a*x < c - b)
    max_x = 10**9  # Large initial value
    for a, b, c in constraints:
        if c > b:
            max_x = min(max_x, (c - b - 1) // a)
        else:
            return 0  # No solution possible
    
    if max_x < 1:
        return 0
    
    # Count valid pairs
    count = 0
    for x in range(1, max_x + 1):
        max_y = 10**9  # Large initial value
        valid = True
        for a, b, c in constraints:
            # Need a*x + b*y < c, so b*y < c - a*x
            if c > a * x:
                max_y = min(max_y, (c - a * x - 1) // b)
            else:
                valid = False
                break
        
        if valid and max_y >= 1:
            count += max_y
    
    return count

t = int(input())
for _ in range(t):
    print(solve_case())