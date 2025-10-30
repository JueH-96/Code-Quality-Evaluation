def solve():
    N, M = map(int, input().split())
    boxes = []
    for i in range(N):
        V, P = map(int, input().split())
        boxes.append((V, P, i))
    
    # Only consider profitable boxes
    profitable = [(V, P, i) for V, P, i in boxes if V > P]
    
    max_profit = 0
    
    # Try all subsets of profitable boxes
    for mask in range(1 << len(profitable)):
        if mask == 0:
            continue
            
        selected_boxes = []
        total_cost = 0
        
        for i in range(len(profitable)):
            if mask & (1 << i):
                V, P, idx = profitable[i]
                selected_boxes.append(V)
                total_cost += P
        
        if not selected_boxes:
            continue
            
        # Calculate balls that can be placed
        k = len(selected_boxes)
        types_used = min(M, k)
        
        # Sort capacities in descending order
        selected_boxes.sort(reverse=True)
        
        if types_used == k:
            # Each box gets its own type
            balls_placed = sum(selected_boxes)
        else:
            # Some boxes must share types
            # Mr. Ball will try to minimize balls placed
            # Heuristic: distribute types to minimize total capacity
            if types_used == 1:
                balls_placed = max(selected_boxes)
            else:
                # When M < k, Mr. Ball can limit efficiency
                # Estimate based on how types can be distributed
                total_capacity = sum(selected_boxes)
                avg_capacity = total_capacity // types_used
                balls_placed = min(total_capacity, types_used * avg_capacity)
        
        profit = balls_placed - total_cost
        max_profit = max(max_profit, profit)
    
    return max_profit

T = int(input())
for _ in range(T):
    print(solve())