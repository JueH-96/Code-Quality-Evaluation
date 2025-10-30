def main():
    import sys
    data = sys.stdin.read().splitlines()
    
    idx = 0
    H_A, W_A = map(int, data[idx].split())
    idx += 1
    A = []
    for _ in range(H_A):
        A.append(data[idx].strip())
        idx += 1
        
    H_B, W_B = map(int, data[idx].split())
    idx += 1
    B = []
    for _ in range(H_B):
        B.append(data[idx].strip())
        idx += 1
        
    H_X, W_X = map(int, data[idx].split())
    idx += 1
    X = []
    for _ in range(H_X):
        X.append(data[idx].strip())
        idx += 1
        
    black_A = set()
    for i in range(H_A):
        for j in range(W_A):
            if A[i][j] == '#':
                black_A.add((i, j))
                
    black_B = set()
    for i in range(H_B):
        for j in range(W_B):
            if B[i][j] == '#':
                black_B.add((i, j))
                
    T = set()
    for i in range(H_X):
        for j in range(W_X):
            if X[i][j] == '#':
                T.add((i, j))
                
    if len(black_A) == 0 or len(black_B) == 0:
        print("No")
        return
        
    min_i_A = min(i for i, j in black_A)
    max_i_A = max(i for i, j in black_A)
    min_j_A = min(j for i, j in black_A)
    max_j_A = max(j for i, j in black_A)
    
    dxA_min = -min_i_A
    dxA_max = H_X - 1 - max_i_A
    dyA_min = -min_j_A
    dyA_max = W_X - 1 - max_j_A
    
    min_i_B = min(i for i, j in black_B)
    max_i_B = max(i for i, j in black_B)
    min_j_B = min(j for i, j in black_B)
    max_j_B = max(j for i, j in black_B)
    
    dxB_min = -min_i_B
    dxB_max = H_X - 1 - max_i_B
    dyB_min = -min_j_B
    dyB_max = W_X - 1 - max_j_B
    
    if dxA_min > dxA_max or dyA_min > dyA_max or dxB_min > dxB_max or dyB_min > dyB_max:
        pass
    else:
        for dxA in range(dxA_min, dxA_max + 1):
            for dyA in range(dyA_min, dyA_max + 1):
                shifted_A = set()
                for (i, j) in black_A:
                    ni = i + dxA
                    nj = j + dyA
                    shifted_A.add((ni, nj))
                    
                for dxB in range(dxB_min, dxB_max + 1):
                    for dyB in range(dyB_min, dyB_max + 1):
                        shifted_B = set()
                        for (i, j) in black_B:
                            ni = i + dxB
                            nj = j + dyB
                            shifted_B.add((ni, nj))
                            
                        union_set = shifted_A | shifted_B
                        if union_set == T:
                            print("Yes")
                            return
                            
    print("No")

if __name__ == "__main__":
    main()