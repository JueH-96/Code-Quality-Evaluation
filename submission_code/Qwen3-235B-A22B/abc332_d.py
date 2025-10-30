import sys
from itertools import permutations

def count_inversions(perm):
    inv = 0
    n = len(perm)
    for i in range(n):
        for j in range(i + 1, n):
            if perm[i] > perm[j]:
                inv += 1
    return inv

def main():
    H, W = map(int, sys.stdin.readline().split())
    A = [list(map(int, sys.stdin.readline().split())) for _ in range(H)]
    B = [list(map(int, sys.stdin.readline().split())) for _ in range(H)]
    
    row_perms = []
    for perm in permutations(range(H)):
        inv = count_inversions(perm)
        row_perms.append((perm, inv))
    
    col_perms = []
    for perm in permutations(range(W)):
        inv = count_inversions(perm)
        col_perms.append((perm, inv))
    
    min_swaps = float('inf')
    
    for row_perm, row_inv in row_perms:
        for col_perm, col_inv in col_perms:
            match = True
            for i in range(H):
                for j in range(W):
                    if A[row_perm[i]][col_perm[j]] != B[i][j]:
                        match = False
                        break
                if not match:
                    break
            if match:
                total = row_inv + col_inv
                if total < min_swaps:
                    min_swaps = total
    
    print(-1 if min_swaps == float('inf') else min_swaps)

if __name__ == '__main__':
    main()