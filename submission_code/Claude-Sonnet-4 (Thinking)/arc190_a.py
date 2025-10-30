def solve():
    N, M = map(int, input().split())
    operations = []
    for _ in range(M):
        L, R = map(int, input().split())
        operations.append((L, R))
    
    covered = [False] * (N + 1)  # 1-indexed
    assignment = [0] * M
    
    def count_uncovered():
        return sum(1 for i in range(1, N+1) if not covered[i])
    
    while count_uncovered() > 0:
        best_op = -1
        best_type = -1
        best_gain = 0
        
        for i in range(M):
            if assignment[i] != 0:
                continue
            
            L, R = operations[i]
            
            # Try operation 1: covers [L, R]
            gain1 = sum(1 for j in range(L, R+1) if not covered[j])
            if gain1 > best_gain:
                best_gain = gain1
                best_op = i
                best_type = 1
            
            # Try operation 2: covers everything outside [L, R]
            gain2 = sum(1 for j in range(1, N+1) if (j < L or j > R) and not covered[j])
            if gain2 > best_gain:
                best_gain = gain2
                best_op = i
                best_type = 2
        
        if best_gain == 0:
            print(-1)
            return
        
        assignment[best_op] = best_type
        L, R = operations[best_op]
        
        if best_type == 1:
            for j in range(L, R+1):
                covered[j] = True
        else:
            for j in range(1, N+1):
                if j < L or j > R:
                    covered[j] = True
    
    cost = sum(1 for x in assignment if x != 0)
    print(cost)
    print(*assignment)

solve()