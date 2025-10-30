# YOUR CODE HERE
import sys
import itertools

def inversion_count(perm):
    inv = 0
    n = len(perm)
    for i in range(n):
        for j in range(i+1,n):
            if perm[i] > perm[j]:
                inv +=1
    return inv

def main():
    import sys
    input = sys.stdin.readline
    H, W = map(int, sys.stdin.readline().split())
    A = [list(map(int, sys.stdin.readline().split())) for _ in range(H)]
    B = [list(map(int, sys.stdin.readline().split())) for _ in range(H)]

    from itertools import permutations

    row_indices = list(range(H))
    col_indices = list(range(W))
    min_swaps = None

    row_perms = []
    for perm in permutations(row_indices):
        inv_count = inversion_count(perm)
        row_perms.append( (perm, inv_count))

    col_perms = []
    for perm in permutations(col_indices):
        inv_count = inversion_count(perm)
        col_perms.append( (perm, inv_count))

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
                total_swaps = row_inv + col_inv
                if min_swaps is None or total_swaps < min_swaps:
                    min_swaps = total_swaps
    if min_swaps is not None:
        print(min_swaps)
    else:
        print(-1)
    
if __name__ == "__main__":
    main()