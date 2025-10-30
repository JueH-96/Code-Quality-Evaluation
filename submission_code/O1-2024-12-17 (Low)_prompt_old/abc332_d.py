def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    H = int(input_data[0])
    W = int(input_data[1])
    
    # Read A
    idx = 2
    A = []
    for _ in range(H):
        row = list(map(int, input_data[idx:idx+W]))
        A.append(row)
        idx += W
    
    # Read B
    B = []
    for _ in range(H):
        row = list(map(int, input_data[idx:idx+W]))
        B.append(row)
        idx += W
    
    # If A is already B, answer is 0
    if A == B:
        print(0)
        return
    
    # Generate all permutations of rows and columns
    from itertools import permutations
    
    def count_inversions(perm):
        # Count the number of inversions in a permutation
        # Using a simple O(n^2) method since n <= 5
        inv = 0
        for i in range(len(perm)):
            for j in range(i+1, len(perm)):
                if perm[i] > perm[j]:
                    inv += 1
        return inv
    
    # Prepare B as a lookup: we want B[row][col] = value
    # We will try each row permutation p and column permutation q,
    # then see if applying them to A gives B.
    
    # Precompute all permutations
    row_perms = list(permutations(range(H)))
    col_perms = list(permutations(range(W)))
    
    min_ops = None
    
    # Convert A to a list of lists for convenience
    A_list = A
    
    for rp in row_perms:
        for cp in col_perms:
            # Apply row perm and col perm to A
            # First reorder rows according to rp, then reorder columns according to cp
            # We'll reconstruct the resulting matrix in a single pass:
            
            # newA[i][j] = A_list[rp[i]][cp[j]]
            # We'll compare it with B[i][j]
            
            # Instead of building the entire matrix fully, we can compare on the fly,
            # but let's just build it to keep it clean and straightforward. 
            
            # Then if it matches B, we compute cost = inversions(rp) + inversions(cp).
            
            candidate = [[A_list[rp[i]][cp[j]] for j in range(W)] for i in range(H)]
            
            if candidate == B:
                cost = count_inversions(rp) + count_inversions(cp)
                if min_ops is None or cost < min_ops:
                    min_ops = cost
    
    if min_ops is None:
        print(-1)
    else:
        print(min_ops)

def main():
    solve()

if __name__ == "__main__":
    main()