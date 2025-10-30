def main():
    import sys
    from itertools import permutations
    
    data = sys.stdin.read().split()
    if not data:
        return
    index = 0
    H = int(data[index])
    index += 1
    W = int(data[index])
    index += 1
    
    # Read grid A
    A = []
    for _ in range(H):
        row = []
        for _ in range(W):
            row.append(int(data[index]))
            index += 1
        A.append(row)
        
    # Read grid B
    B = []
    for _ in range(H):
        row = []
        for _ in range(W):
            row.append(int(data[index]))
            index += 1
        B.append(row)
    
    # Helper function to compute number of inversions in a permutation,
    # which is equal to the minimal number of adjacent swaps needed to realize it.
    def inversion_count(perm):
        inv = 0
        n = len(perm)
        for i in range(n):
            for j in range(i + 1, n):
                if perm[i] > perm[j]:
                    inv += 1
        return inv
    
    best = float('inf')
    
    # Try every permutation for rows and columns.
    for pr in permutations(range(H)):
        row_ops = inversion_count(pr)
        for pc in permutations(range(W)):
            col_ops = inversion_count(pc)
            valid = True
            # Check if the grid A transformed by the two permutations equals grid B.
            for i in range(H):
                for j in range(W):
                    if A[pr[i]][pc[j]] != B[i][j]:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                best = min(best, row_ops + col_ops)
    
    if best == float('inf'):
        sys.stdout.write(str(-1))
    else:
        sys.stdout.write(str(best))

if __name__ == '__main__':
    main()