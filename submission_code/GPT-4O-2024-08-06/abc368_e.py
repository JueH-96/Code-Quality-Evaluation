def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    X1 = int(data[2])
    
    trains = []
    index = 3
    for _ in range(M):
        A = int(data[index])
        B = int(data[index + 1])
        S = int(data[index + 2])
        T = int(data[index + 3])
        trains.append((A, B, S, T))
        index += 4
    
    # Initialize X array
    X = [float('inf')] * M
    X[0] = X1  # X1 is given
    
    # Bellman-Ford style relaxation
    for _ in range(M - 1):  # M-1 relaxations
        for i in range(M):
            A_i, B_i, S_i, T_i = trains[i]
            for j in range(M):
                A_j, B_j, S_j, T_j = trains[j]
                if B_i == A_j and T_i <= S_j:
                    # Relax the edge i -> j
                    if X[i] + (S_j - T_i) < X[j]:
                        X[j] = X[i] + (S_j - T_i)
    
    # Output the result
    print(' '.join(map(str, X[1:])))  # We need X_2, X_3, ..., X_M