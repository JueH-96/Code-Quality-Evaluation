def solve():
    N, X = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    P = list(map(int, input().split()))
    Q = list(map(int, input().split()))
    
    # Convert to 0-indexed
    X -= 1
    P = [p - 1 for p in P]
    Q = [q - 1 for q in Q]
    
    # Check if already solved
    total_balls = sum(A) + sum(B)
    if total_balls == A[X] + B[X]:
        print(0)
        return
    
    # Build reverse graph to find boxes that can reach X
    can_reach_X = [False] * N
    can_reach_X[X] = True
    
    # BFS/DFS to find all boxes that can reach X
    changed = True
    while changed:
        changed = False
        for i in range(N):
            if not can_reach_X[i]:
                if can_reach_X[P[i]] or can_reach_X[Q[i]]:
                    can_reach_X[i] = True
                    changed = True
    
    # Check if all boxes with balls can reach X
    for i in range(N):
        if (A[i] > 0 or B[i] > 0) and not can_reach_X[i]:
            print(-1)
            return
    
    # Count operations needed
    operations = 0
    current_A = A[:]
    current_B = B[:]
    
    # Keep operating until all balls are in box X
    while True:
        # Check if done
        done = True
        for i in range(N):
            if i != X and (current_A[i] > 0 or current_B[i] > 0):
                done = False
                break
        if done:
            break
        
        # Find boxes to operate on
        to_operate = []
        for i in range(N):
            if i != X and (current_A[i] > 0 or current_B[i] > 0):
                to_operate.append(i)
        
        # Operate on all boxes that need it
        new_A = current_A[:]
        new_B = current_B[:]
        
        for i in to_operate:
            if current_A[i] > 0:
                new_A[i] -= current_A[i]
                new_A[P[i]] += current_A[i]
            if current_B[i] > 0:
                new_B[i] -= current_B[i]
                new_B[Q[i]] += current_B[i]
            operations += 1
        
        current_A = new_A
        current_B = new_B
    
    print(operations)

solve()