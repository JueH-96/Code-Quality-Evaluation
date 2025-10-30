def solve():
    N, M = map(int, input().split())
    operations = []
    for _ in range(M):
        L, R = map(int, input().split())
        operations.append((L, R))
    
    # For each position, track which operations can cover it
    can_cover = [[] for _ in range(N + 1)]
    for i in range(M):
        L, R = operations[i]
        # Type 1: covers [L, R]
        for j in range(L, R + 1):
            if j <= N:
                can_cover[j].append((i, 1))
        # Type 2: covers [1, L-1] and [R+1, N]
        for j in range(1, L):
            if j <= N:
                can_cover[j].append((i, 2))
        for j in range(R + 1, N + 1):
            if j <= N:
                can_cover[j].append((i, 2))
    
    # Check if every position can be covered
    for j in range(1, N + 1):
        if not can_cover[j]:
            print(-1)
            return
    
    # Greedy algorithm
    covered = [False] * (N + 1)
    chosen_ops = [0] * M
    
    while True:
        # Check if all positions are covered
        all_covered = True
        for j in range(1, N + 1):
            if not covered[j]:
                all_covered = False
                break
        
        if all_covered:
            break
        
        # Find best operation to use next
        best_op = -1
        best_type = -1
        best_score = -1
        
        for i in range(M):
            if chosen_ops[i] != 0:
                continue
                
            L, R = operations[i]
            
            # Count new positions covered by type 1
            new_covered_1 = 0
            for j in range(L, R + 1):
                if j <= N and not covered[j]:
                    new_covered_1 += 1
            
            # Count new positions covered by type 2
            new_covered_2 = 0
            for j in range(1, L):
                if j <= N and not covered[j]:
                    new_covered_2 += 1
            for j in range(R + 1, N + 1):
                if j <= N and not covered[j]:
                    new_covered_2 += 1
            
            # Choose the option that covers more new positions
            if new_covered_1 > best_score:
                best_op = i
                best_type = 1
                best_score = new_covered_1
            
            if new_covered_2 > best_score:
                best_op = i
                best_type = 2
                best_score = new_covered_2
        
        if best_score == 0:
            print(-1)
            return
        
        # Apply the chosen operation
        chosen_ops[best_op] = best_type
        L, R = operations[best_op]
        
        if best_type == 1:
            for j in range(L, R + 1):
                if j <= N:
                    covered[j] = True
        else:
            for j in range(1, L):
                if j <= N:
                    covered[j] = True
            for j in range(R + 1, N + 1):
                if j <= N:
                    covered[j] = True
    
    # Output result
    cost = sum(1 for op in chosen_ops if op != 0)
    print(cost)
    print(' '.join(map(str, chosen_ops)))

solve()