def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    S = data[N]
    
    # Precompute the positions of M, E, X
    M_indices = [i for i, c in enumerate(S) if c == 'M']
    E_indices = [i for i, c in enumerate(S) if c == 'E']
    X_indices = [i for i, c in enumerate(S) if c == 'X']
    
    # Precompute the counts of A[i] for E and X
    # For each E, we need to know the count of A[j] for j in E_indices
    # Similarly for X
    
    # To optimize, we can precompute the counts of A[j] for E and A[k] for X
    # But since we need to consider the order i < j < k, we need to process in order
    
    # We will iterate over all possible M, then for each M, find all E after it, then for each E, find all X after it
    
    # To make it efficient, we can precompute the positions of E and X in a way that allows us to quickly find the next E or X after a given index
    
    # First, create a list of E and X indices in order
    E_sorted = sorted(E_indices)
    X_sorted = sorted(X_indices)
    
    # Precompute the mex for all possible combinations of A_i, A_j, A_k
    # Since A_i, A_j, A_k are in {0,1,2}, the mex can be precomputed for all 27 combinations
    mex_table = {}
    for a in range(3):
        for b in range(3):
            for c in range(3):
                s = {a, b, c}
                mex = 0
                while mex in s:
                    mex += 1
                mex_table[(a, b, c)] = mex
    
    total = 0
    
    # Iterate over all M positions
    for i in M_indices:
        a_i = A[i]
        # Find all E positions after i
        # Use binary search to find the first E index greater than i
        low = 0
        high = len(E_sorted) - 1
        first_E = -1
        while low <= high:
            mid = (low + high) // 2
            if E_sorted[mid] > i:
                first_E = mid
                high = mid - 1
            else:
                low = mid + 1
        if first_E == -1:
            continue
        # Iterate over all E positions after i
        for j_idx in range(first_E, len(E_sorted)):
            j = E_sorted[j_idx]
            a_j = A[j]
            # Find all X positions after j
            # Use binary search to find the first X index greater than j
            low_x = 0
            high_x = len(X_sorted) - 1
            first_X = -1
            while low_x <= high_x:
                mid_x = (low_x + high_x) // 2
                if X_sorted[mid_x] > j:
                    first_X = mid_x
                    high_x = mid_x - 1
                else:
                    low_x = mid_x + 1
            if first_X == -1:
                continue
            # Iterate over all X positions after j
            for k_idx in range(first_X, len(X_sorted)):
                k = X_sorted[k_idx]
                a_k = A[k]
                # Compute mex
                mex = mex_table[(a_i, a_j, a_k)]
                total += mex
    
    print(total)

if __name__ == "__main__":
    main()