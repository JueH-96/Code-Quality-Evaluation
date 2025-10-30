import sys
import itertools

def main():
    # Read H and W
    H, W = map(int, sys.stdin.readline().split())
    
    # Read grid A
    A = []
    for _ in range(H):
        A.append(list(map(int, sys.stdin.readline().split())))
    
    # Read grid B
    B = []
    for _ in range(H):
        B.append(list(map(int, sys.stdin.readline().split())))
    
    # Generate all permutations of rows and columns
    row_perms = list(itertools.permutations(range(H)))
    col_perms = list(itertools.permutations(range(W)))
    
    min_ops = float('inf')
    
    # Iterate over all pairs of row and column permutations
    for P in row_perms:
        for Q in col_perms:
            # Check if A[P[i]][Q[j]] == B[i][j] for all i, j
            matches = True
            for i in range(H):
                for j in range(W):
                    if A[P[i]][Q[j]] != B[i][j]:
                        matches = False
                        break
                if not matches:
                    break
            if matches:
                # Compute inversions in P and Q
                ops = count_inversions(P) + count_inversions(Q)
                if ops < min_ops:
                    min_ops = ops
    
    if min_ops < float('inf'):
        print(min_ops)
    else:
        print(-1)

def count_inversions(perm):
    count = 0
    n = len(perm)
    for i in range(n):
        for j in range(i+1, n):
            if perm[i] > perm[j]:
                count += 1
    return count

if __name__ == '__main__':
    main()