import sys, itertools

def inversion_count(perm):
    count = 0
    n = len(perm)
    for i in range(n):
        for j in range(i+1, n):
            if perm[i] > perm[j]:
                count += 1
    return count

def main():
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    H = int(next(it))
    W = int(next(it))
    A = []
    for _ in range(H):
        row = []
        for _ in range(W):
            row.append(int(next(it)))
        A.append(row)
    B = []
    for _ in range(H):
        row = []
        for _ in range(W):
            row.append(int(next(it)))
        B.append(row)
        
    best = None
    
    # We can perform arbitrary reordering of rows and columns.
    # The minimum number of operations needed to achieve a given permutation is the inversion count of the permutation.
    # We try all possible row and column permutations and check if they transform A to B.
    for row_perm in itertools.permutations(range(H)):
        for col_perm in itertools.permutations(range(W)):
            valid = True
            for i in range(H):
                for j in range(W):
                    if A[row_perm[i]][col_perm[j]] != B[i][j]:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                cost = inversion_count(row_perm) + inversion_count(col_perm)
                if best is None or cost < best:
                    best = cost

    if best is None:
        print(-1)
    else:
        print(best)

if __name__ == '__main__':
    main()