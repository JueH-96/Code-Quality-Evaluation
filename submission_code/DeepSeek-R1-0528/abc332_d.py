import itertools
import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    
    H = int(data[0])
    W = int(data[1])
    idx = 2
    A = []
    for _ in range(H):
        row = list(map(int, data[idx:idx+W]))
        idx += W
        A.append(row)
    
    B = []
    for _ in range(H):
        row = list(map(int, data[idx:idx+W]))
        idx += W
        B.append(row)
    
    flat_A = [num for row in A for num in row]
    flat_B = [num for row in B for num in row]
    if sorted(flat_A) != sorted(flat_B):
        print(-1)
        return
    
    row_perms = list(itertools.permutations(range(H)))
    col_perms = list(itertools.permutations(range(W)))
    min_ops = float('inf')
    
    for r_perm in row_perms:
        for c_perm in col_perms:
            valid = True
            for i in range(H):
                for j in range(W):
                    if A[r_perm[i]][c_perm[j]] != B[i][j]:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                inv_r = 0
                n = len(r_perm)
                for i in range(n):
                    for j in range(i+1, n):
                        if r_perm[i] > r_perm[j]:
                            inv_r += 1
                
                inv_c = 0
                m = len(c_perm)
                for i in range(m):
                    for j in range(i+1, m):
                        if c_perm[i] > c_perm[j]:
                            inv_c += 1
                
                total_ops = inv_r + inv_c
                if total_ops < min_ops:
                    min_ops = total_ops
    
    print(min_ops if min_ops != float('inf') else -1)

if __name__ == "__main__":
    main()