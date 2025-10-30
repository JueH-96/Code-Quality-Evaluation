import sys
from itertools import permutations

def inversion_count(p):
    cnt = 0
    n = len(p)
    for i in range(n):
        for j in range(i+1, n):
            if p[i] > p[j]:
                cnt += 1
    return cnt

def main():
    H, W = map(int, sys.stdin.readline().split())
    A = [list(map(int, sys.stdin.readline().split())) for _ in range(H)]
    B = [list(map(int, sys.stdin.readline().split())) for _ in range(H)]
    
    min_ops = float('inf')
    
    # Generate all possible row permutations
    for row_perm in permutations(range(H)):
        row_inv = inversion_count(row_perm)
        # For each row permutation, generate all column permutations
        for col_perm in permutations(range(W)):
            col_inv = inversion_count(col_perm)
            total = row_inv + col_inv
            # Check if applying row_perm and col_perm to A gives B
            valid = True
            for i in range(H):
                for j in range(W):
                    if A[row_perm[i]][col_perm[j]] != B[i][j]:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                if total < min_ops:
                    min_ops = total
    if min_ops == float('inf'):
        print(-1)
    else:
        print(min_ops)

if __name__ == '__main__':
    main()