def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    X = int(data[1])
    
    U = []
    D = []
    
    index = 2
    for _ in range(N):
        U.append(int(data[index]))
        D.append(int(data[index + 1]))
        index += 2
    
    # Calculate the target H
    S = [U[i] + D[i] for i in range(N)]
    H = max(S)
    
    # Calculate the cost to make U_i + D_i = H
    cost = 0
    for i in range(N):
        target_U_i = H - D[i]
        if U[i] > target_U_i:
            cost += U[i] - target_U_i
            U[i] = target_U_i
    
    # Ensure |U_i - U_{i+1}| <= X
    for i in range(N - 1):
        if abs(U[i] - U[i + 1]) > X:
            if U[i] > U[i + 1]:
                excess = U[i] - U[i + 1] - X
                cost += excess
                U[i] -= excess
            else:
                excess = U[i + 1] - U[i] - X
                cost += excess
                U[i + 1] -= excess
    
    print(cost)