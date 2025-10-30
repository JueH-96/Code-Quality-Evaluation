# YOUR CODE HERE
def solve_case():
    n = int(input())
    constraints = []
    for _ in range(n):
        a, b, c = map(int, input().split())
        constraints.append((a, b, c))
    
    # Find maximum possible x
    max_x = float('inf')
    for a, b, c in constraints:
        if c <= b:  # No valid x exists for this constraint
            return 0
        # For y = 1, we need a*x + b < c, so x < (c-b)/a
        max_x = min(max_x, (c - b - 1) // a)
    
    if max_x < 1:
        return 0
    
    count = 0
    # For each valid x, find the maximum valid y
    for x in range(1, max_x + 1):
        max_y = float('inf')
        for a, b, c in constraints:
            # We need a*x + b*y < c
            # So b*y < c - a*x
            # y < (c - a*x) / b
            if c <= a * x:  # No valid y for this x and constraint
                max_y = 0
                break
            max_y = min(max_y, (c - a * x - 1) // b)
        
        if max_y >= 1:
            count += max_y
    
    return count

# Main execution
t = int(input())
for _ in range(t):
    print(solve_case())