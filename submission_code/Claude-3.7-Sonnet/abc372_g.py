def solve():
    N = int(input())
    constraints = []
    for _ in range(N):
        A, B, C = map(int, input().split())
        constraints.append((A, B, C))
    
    # Find the maximum value of x
    max_x = float('inf')
    for A, B, C in constraints:
        # For A*x + B*y < C with y >= 1, we have A*x + B < C, or x < (C - B) / A
        if C <= B:
            return 0
        
        # The maximum valid x is the largest integer strictly less than (C - B) / A
        if (C - B) % A == 0:
            curr_max_x = (C - B) // A - 1
        else:
            curr_max_x = (C - B) // A
        
        max_x = min(max_x, curr_max_x)
    
    # If max_x < 1, then no valid pairs
    if max_x < 1:
        return 0
    
    # Count valid pairs
    count = 0
    for x in range(1, max_x + 1):
        min_valid_y_bound = float('inf')
        for A, B, C in constraints:
            # A*x + B*y < C, so y < (C - A*x) / B
            # The maximum valid y is the largest positive integer strictly less than (C - A*x) / B
            remaining = C - A * x
            if remaining <= 0:
                min_valid_y_bound = 0
                break
            
            if remaining % B == 0:
                valid_y_bound = remaining // B - 1
            else:
                valid_y_bound = remaining // B
            
            min_valid_y_bound = min(min_valid_y_bound, valid_y_bound)
        
        # Add the number of valid y for this x (which is min_valid_y_bound)
        count += max(min_valid_y_bound, 0)
    
    return count

T = int(input())
for _ in range(T):
    print(solve())