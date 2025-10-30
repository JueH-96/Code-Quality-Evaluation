def main():
    import sys
    input = sys.stdin.readline

    N = int(input())
    A = list(map(int, input().split()))
    
    # Dictionary to store positions (0-indexed) for each value.
    pos_dict = {}
    for i, a in enumerate(A):
        if a not in pos_dict:
            pos_dict[a] = []
        pos_dict[a].append(i)
        
    result = 0
    # For every distinct value in A, we will calculate the contribution from triples
    # of the form (i, j, k) where A[i] = A[k] = x.
    # For a given value x, assume positions are stored in a list pos (sorted).
    # For every pair of positions (i, k) in pos with i < k, count j's with:
    #    i < j < k and A[j] != x.
    # The total indices between i and k is (k - i - 1). However, some of these indices
    # might have A[j] = x. Specifically, if there are m occurrences of x between i and k
    # (which is (pos_index_difference - 1) if using positions list), then valid choices for j are:
    #    k - i - 1 - (m), where m = (k_index - i_index - 1)
    #
    # Therefore, the count for a pair of positions (i, k) becomes:
    #    (k - i - 1) - (k_index - i_index - 1) = (k - i) - (k_index - i_index)
    #
    # Summing over all pairs in the positions list for x is inefficient.
    # Instead, using an algebraic transformation we can derive that for a sorted
    # positions list, if we denote pos[s] as the s-th occurrence (0-indexed)
    # then the contribution of all pairs ending at index pos[s] for s>=1 is:
    #
    #   Sum_{r=0}^{s-1} ( (pos[s] - s) - (pos[r] - r) - 1 )
    #
    # We can compute this quickly by maintaining a prefix sum for (pos[r] - r).
    # Then for each s, the contribution is:
    #   s*(pos[s] - s) - prefix[s-1] - s
    #
    # and we add these contributions over all s from 1 to len(pos)-1.
    
    for positions in pos_dict.values():
        if len(positions) < 2:
            continue
        positions.sort()  # Ensure positions are sorted (they are in order but safety check)
        prefix_sum = 0
        L = len(positions)
        for s in range(L):
            if s > 0:
                # Calculate contributions from each pair ending at positions[s]
                result += s * (positions[s] - s) - prefix_sum - s
            prefix_sum += positions[s] - s
            
    sys.stdout.write(str(result))
    
if __name__ == '__main__':
    main()