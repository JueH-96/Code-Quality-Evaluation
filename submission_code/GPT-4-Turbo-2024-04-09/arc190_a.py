def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    operations = []
    index = 2
    for _ in range(M):
        L = int(data[index])
        R = int(data[index + 1])
        operations.append((L, R))
        index += 2
    
    if M == 0:
        print(-1)
        return
    
    # We need to cover all positions from 1 to N with 1s.
    # We can use a greedy approach to find the minimum number of operations to cover all positions.
    
    # We can use a difference array technique to mark the ranges efficiently
    diff = [0] * (N + 2)  # We use N+2 to avoid boundary issues when marking end+1
    
    # Mark all operations as no-op initially
    op_types = [0] * M
    
    # We need to find the minimal set of operations that can set all x_i to 1
    # We can achieve this by ensuring that the entire range [1, N] is covered by operation 1 or 2
    
    # To minimize the cost, we should try to use as few operations as possible
    # Strategy:
    # 1. Use operation 1 on the widest possible range that covers [1, N] if possible
    # 2. If not possible, check if we can use operation 2 to set the outer parts and operation 1 for the rest
    
    # First, check if we can cover [1, N] directly
    full_cover_possible = False
    for i, (L, R) in enumerate(operations):
        if L == 1 and R == N:
            full_cover_possible = True
            op_types[i] = 1  # Use operation 1 on this operation
            break
    
    if full_cover_possible:
        print(1)
        print(" ".join(map(str, op_types)))
        return
    
    # If not directly possible, we need to check the combination of operations
    # We need to find the minimal segments that can cover [1, N] when combined
    
    # We will use a greedy approach to find the minimal number of segments to cover [1, N]
    # We will sort the operations by L and try to extend the coverage as far as possible
    
    operations.sort()
    
    # To cover from 1 to N
    current_end = 0
    min_cost = 0
    selected_operations = []
    
    i = 0
    while i < M:
        if current_end >= N:
            break
        # Find the operation that starts at or before current_end + 1 and extends as far as possible
        best_extension = current_end
        best_index = -1
        while i < M and operations[i][0] <= current_end + 1:
            if operations[i][1] > best_extension:
                best_extension = operations[i][1]
                best_index = i
            i += 1
        if best_index == -1:
            # No operation can extend the coverage
            print(-1)
            return
        # Use this operation
        selected_operations.append(best_index)
        current_end = best_extension
        min_cost += 1
    
    if current_end < N:
        print(-1)
        return
    
    # We have a valid coverage, set the operations
    for idx in selected_operations:
        op_types[idx] = 1  # Use operation 1 for selected operations
    
    print(min_cost)
    print(" ".join(map(str, op_types)))