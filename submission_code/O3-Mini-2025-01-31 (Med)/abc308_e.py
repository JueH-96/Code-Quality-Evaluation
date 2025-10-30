def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    it = iter(input_data)
    N = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    S = next(it).strip()
    
    # Precompute the count of indices with S[i]=='X' for each A[i] value.
    suffix_counts_x = [0, 0, 0]
    for i in range(N):
        if S[i] == 'X':
            suffix_counts_x[A[i]] += 1

    # prefix_counts_m will store counts for indices with S[i]=='M' as we iterate.
    prefix_counts_m = [0, 0, 0]
    
    # Define a helper function to compute mex. Since A_i can only be 0,1,2,
    # the mex is computed as the smallest non-negative integer not in {a, b, c}.
    def mex(a, b, c):
        s = {a, b, c}
        if 0 not in s:
            return 0
        if 1 not in s:
            return 1
        if 2 not in s:
            return 2
        return 3  # when set is exactly {0, 1, 2}
    
    total = 0
    
    # Iterate through indices considering them as the E position (middle index).
    # We will maintain the prefix counts for 'M' and suffix counts for 'X' ensuring
    # i < j < k. Before processing an index j, if S[j]=='X', remove it from suffix_counts_x.
    for j in range(N):
        # Remove current index j from suffix_counts_x if it is a 'X'
        if S[j] == 'X':
            suffix_counts_x[A[j]] -= 1
        
        # If S[j] is 'E', then count all valid tuples with i < j (M) and k > j (X)
        if S[j] == 'E':
            # For every possible value for M and X, add the contribution:
            # contribution = (number of M's with that value) * (number of X's with that value)
            #                * mex(val_M, A[j], val_X)
            for val_m in range(3):
                if prefix_counts_m[val_m] == 0:
                    continue
                for val_x in range(3):
                    if suffix_counts_x[val_x] == 0:
                        continue
                    total += prefix_counts_m[val_m] * suffix_counts_x[val_x] * mex(val_m, A[j], val_x)
        
        # After processing, update the prefix count if the current letter is 'M'
        if S[j] == 'M':
            prefix_counts_m[A[j]] += 1
    
    sys.stdout.write(str(total))

if __name__ == '__main__':
    main()