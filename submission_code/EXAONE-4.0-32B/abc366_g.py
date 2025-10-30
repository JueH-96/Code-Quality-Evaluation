def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    
    n = int(data[0])
    m = int(data[1])
    A = [[0] * n for _ in range(n)]
    index = 2
    for i in range(m):
        u = int(data[index]); v = int(data[index+1]); index += 2
        u -= 1
        v -= 1
        A[u][v] = 1
        A[v][u] = 1
        
    M = [row[:] for row in A]
    pivot_cols = []
    r = 0
    for col in range(n):
        pivot_row = -1
        for row in range(r, n):
            if M[row][col] == 1:
                pivot_row = row
                break
        if pivot_row == -1:
            continue
            
        M[r], M[pivot_row] = M[pivot_row], M[r]
        
        for i in range(n):
            if i != r and M[i][col] == 1:
                for j in range(col, n):
                    M[i][j] ^= M[r][j]
                    
        pivot_cols.append(col)
        r += 1
        
    d = n - r
    if d == 0:
        print("No")
        return
        
    free_cols = [col for col in range(n) if col not in pivot_cols]
    
    basis = []
    for fc in free_cols:
        vec = [0] * n
        vec[fc] = 1
        for i in range(r):
            if M[i][fc] == 1:
                pc = pivot_cols[i]
                vec[pc] = 1
        basis.append(vec)
        
    covered = [False] * n
    for vec in basis:
        for i in range(n):
            if vec[i] == 1:
                covered[i] = True
                
    if not all(covered):
        print("No")
        return
        
    res = [0] * n
    for k in range(d):
        bit_val = 1 << k
        for i in range(n):
            if basis[k][i] == 1:
                res[i] |= bit_val
                
    print("Yes")
    print(" ".join(str(x) for x in res))

if __name__ == "__main__":
    main()