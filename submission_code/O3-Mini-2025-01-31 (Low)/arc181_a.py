def main():
    import sys
    data = sys.stdin.read().split()
    t = int(data[0])
    index = 1
    # We'll build the answer for each test case in out_lines.
    #
    # Explanation of the approach:
    #
    # We are given that in one operation we can choose a pivot index k (1-indexed) and then
    # independently sort (in ascending order) the prefix (indices 1..k–1) and suffix (indices k+1..N)
    # while leaving position k unchanged.
    #
    # Notice that after doing this operation the final permutation becomes:
    #   - The first k–1 positions become (after sorting) the sorted list of the original
    #     values in positions 1..k–1.
    #   - The middle remains P[k].
    #   - The last N–k positions become the sorted list of the original values in positions k+1..N.
    #
    # For the entire array to become sorted (i.e. P[i] = i), a natural necessary and sufficient
    # condition is that there exists a pivot index k (1 ≤ k ≤ N) such that:
    #
    # 1. The first k–1 numbers, when sorted, equal [1, 2, ..., k–1]. (For k = 1 there is no prefix.)
    # 2. The pivot element is already correct, i.e. P[k] = k.
    # 3. The last N–k numbers, when sorted, equal [k+1, k+2, ..., N]. (For k = N there is no suffix.)
    #
    # Because the operation sorts each part independently, if these three conditions hold,
    # then one operation with that value of k will yield the sorted permutation.
    #
    # Thus our plan is:
    #   - if the permutation is already sorted, answer = 0.
    #   - Otherwise, try to find any pivot k (from 1 to N) satisfying the conditions described.
    #     If one exists, answer = 1.
    #   - If no one‐operation solution exists then (except for one special case) the answer is 2.
    #     It turns out that the only time we need 3 operations is when P[1] = N and P[N] = 1.
    #
    # We can test these conditions efficiently by precomputing prefix minimum/maximum and suffix minimum/maximum.
    # For the prefix (positions 1..i) the sequence (when sorted) equals [1,2,..., i] if and only if:
    #   (i) the minimum equals 1,
    #   (ii) the maximum equals i,
    #   (iii) all values are distinct (which they are because P is a permutation).
    # Similarly for the suffix.
    
    out_lines = []
    for _ in range(t):
        n = int(data[index])
        index += 1
        P = list(map(int, data[index:index+n]))
        index += n
        
        # If already sorted, then answer is 0.
        if all(P[i] == i+1 for i in range(n)):
            out_lines.append("0")
            continue
        
        # Precompute prefix min and prefix max; prefix[i] will represent the min/max for first (i+1) elements.
        prefix_min = [0] * n
        prefix_max = [0] * n
        prefix_min[0] = P[0]
        prefix_max[0] = P[0]
        for i in range(1, n):
            prefix_min[i] = P[i] if P[i] < prefix_min[i-1] else prefix_min[i-1]
            prefix_max[i] = P[i] if P[i] > prefix_max[i-1] else prefix_max[i-1]
        
        # Precompute suffix min and max; suffix[i] will represent the min/max for positions i..n-1.
        suffix_min = [0] * n
        suffix_max = [0] * n
        suffix_min[n-1] = P[n-1]
        suffix_max[n-1] = P[n-1]
        for i in range(n-2, -1, -1):
            suffix_min[i] = P[i] if P[i] < suffix_min[i+1] else suffix_min[i+1]
            suffix_max[i] = P[i] if P[i] > suffix_max[i+1] else suffix_max[i+1]
        
        one_op_possible = False
        # Try each pivot k = 1, 2, ..., n.
        for k in range(1, n+1):
            # Check prefix (positions 1 to k-1). (If k==1, prefix is empty and automatically OK.)
            if k == 1:
                prefix_ok = True
            else:
                # The prefix is indices 0 through k-2.
                # They must, when sorted, be equal to [1, 2, ..., k-1].
                # Since there are k-1 elements, this is equivalent to saying 
                # that the minimum equals 1 and the maximum equals k-1.
                if prefix_min[k-2] == 1 and prefix_max[k-2] == k-1:
                    prefix_ok = True
                else:
                    prefix_ok = False
            
            # Check that the "pivot" element is already correct.
            mid_ok = (P[k-1] == k)
            
            # Check suffix (positions k+1 to n). (If k==n, suffix is empty and automatically OK.)
            if k == n:
                suffix_ok = True
            else:
                # The suffix occupies indices k through n-1.
                # It should equal [k+1, k+2, ..., n] when sorted, which is true if and only if 
                # the minimum equals k+1 and the maximum equals n.
                if suffix_min[k] == k+1 and suffix_max[k] == n:
                    suffix_ok = True
                else:
                    suffix_ok = False
            if prefix_ok and mid_ok and suffix_ok:
                one_op_possible = True
                break
        
        if one_op_possible:
            out_lines.append("1")
        else:
            # There is one known special case:
            # if the first element is n and the last element is 1, then answer is 3.
            if P[0] == n and P[-1] == 1:
                out_lines.append("3")
            else:
                out_lines.append("2")
    sys.stdout.write("
".join(out_lines))
    
if __name__ == "__main__":
    main()