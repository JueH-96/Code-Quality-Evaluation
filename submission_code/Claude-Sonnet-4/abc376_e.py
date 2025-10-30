def solve():
    n, k = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    
    # Create list of (A_i, B_i, index) and sort by A_i
    items = [(A[i], B[i], i) for i in range(n)]
    items.sort()
    
    min_result = float('inf')
    
    # Try each possible maximum A value
    for max_idx in range(k-1, n):  # We need at least k elements
        max_a = items[max_idx][0]
        
        # Get all elements with A_i <= max_a
        candidates = []
        for i in range(max_idx + 1):
            candidates.append((items[i][1], items[i][2]))  # (B_i, original_index)
        
        # Sort candidates by B_i to get smallest B values
        candidates.sort()
        
        # Take the k smallest B values
        selected_b_sum = sum(candidates[i][0] for i in range(k))
        
        # Calculate the expression value
        result = max_a * selected_b_sum
        min_result = min(min_result, result)
    
    return min_result

T = int(input())
for _ in range(T):
    print(solve())