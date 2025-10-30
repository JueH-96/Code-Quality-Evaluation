import sys

def solve_case():
    n = int(input())
    constraints = []
    for _ in range(n):
        a, b, c = map(int, input().split())
        constraints.append((a, b, c))
    
    count = 0
    
    # We need to find a reasonable upper bound for x
    # For each constraint A_i * x + B_i * y < C_i
    # Since y >= 1, we need A_i * x + B_i < C_i
    # So x < (C_i - B_i) / A_i
    max_x = float('inf')
    for a, b, c in constraints:
        if c <= b:  # No valid solution possible for this constraint
            return 0
        max_x = min(max_x, (c - b - 1) // a)
    
    if max_x < 1:
        return 0
    
    # For each valid x, find the maximum valid y
    for x in range(1, int(max_x) + 1):
        max_y = float('inf')
        valid = True
        
        for a, b, c in constraints:
            if a * x >= c:  # No valid y for this constraint
                valid = False
                break
            # B_i * y < C_i - A_i * x
            # y < (C_i - A_i * x) / B_i
            max_y_for_constraint = (c - a * x - 1) // b
            max_y = min(max_y, max_y_for_constraint)
        
        if valid and max_y >= 1:
            count += max_y
    
    return count

def main():
    t = int(input())
    for _ in range(t):
        result = solve_case()
        print(result)

if __name__ == "__main__":
    main()