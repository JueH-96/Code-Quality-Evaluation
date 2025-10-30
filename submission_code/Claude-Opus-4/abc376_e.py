def solve_case():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    
    # Create list of (a_i, b_i, index) tuples
    items = [(a[i], b[i], i) for i in range(n)]
    
    # Sort by a values
    items.sort()
    
    min_result = float('inf')
    
    # Try each possible maximum A value
    for max_idx in range(k-1, n):
        max_a = items[max_idx][0]
        
        # Collect all valid items (with a <= max_a)
        valid_items = []
        for i in range(max_idx + 1):
            valid_items.append(items[i][1])  # Just the b values
        
        # Sort b values and take k smallest
        valid_items.sort()
        sum_b = sum(valid_items[:k])
        
        result = max_a * sum_b
        min_result = min(min_result, result)
    
    return min_result

# Read number of test cases
t = int(input())

# Process each test case
for _ in range(t):
    print(solve_case())