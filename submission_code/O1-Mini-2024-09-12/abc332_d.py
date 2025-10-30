import sys
import itertools

def count_inversions(p):
    inv = 0
    n = len(p)
    for i in range(n):
        for j in range(i+1, n):
            if p[i] > p[j]:
                inv += 1
    return inv

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    H = int(data[idx]); idx +=1
    W = int(data[idx]); idx +=1
    A = []
    for _ in range(H):
        row = list(map(int, data[idx:idx+W]))
        A.append(row)
        idx += W
    B = []
    for _ in range(H):
        row = list(map(int, data[idx:idx+W]))
        B.append(row)
        idx += W
    row_perms = list(itertools.permutations(range(H)))
    col_perms = list(itertools.permutations(range(W)))
    # Precompute inversion counts
    row_inv = {perm: count_inversions(perm) for perm in row_perms}
    col_inv = {perm: count_inversions(perm) for perm in col_perms}
    min_ops = float('inf')
    for r_perm in row_perms:
        # Apply row permutation
        A_rows = [A[r_perm[i]] for i in range(H)]
        # Now, for columns, find permutation that makes columns match
        # Iterate all column permutations
        for c_perm in col_perms:
            match = True
            for i in range(H):
                for j in range(W):
                    if A_rows[i][c_perm[j]] != B[i][j]:
                        match = False
                        break
                if not match:
                    break
            if match:
                total_ops = row_inv[r_perm] + col_inv[c_perm]
                if total_ops < min_ops:
                    min_ops = total_ops
    if min_ops == float('inf'):
        print(-1)
    else:
        print(min_ops)

if __name__ == "__main__":
    main()