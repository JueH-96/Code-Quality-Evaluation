def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    t = int(input_data[0])
    idx = 1
    
    # A fast way to compute answers for all test cases
    # under the given constraints.
    #
    # Key observations:
    # 1) If P is already the identity permutation (P_i = i), answer is 0.
    # 2) Otherwise, check if there exists k in [1..N] so that:
    #       - The subarray P[1..k-1] contains exactly the numbers 1..k-1.
    #       - The element P[k] is k.
    #       - The subarray P[k+1..N] contains exactly the numbers k+1..N.
    #    If such k exists, we can fix the permutation in one operation by picking that k.
    #    Otherwise, the answer is 2.
    #
    # To check the subarray conditions efficiently:
    #  - prefix_max[i] = max of P[0..i]  (0-based)
    #  - suffix_min[i] = min of P[i..N-1] (0-based)
    #
    # For a candidate k (1-based):
    #  - if k > 1, we need prefix_max[k-2] == k-1
    #  - if k < N, we need suffix_min[k] == k+1
    #  - P[k-1] == k
    #
    # Implementation is O(N) per test because we only build prefix/suffix arrays
    # and do a single pass over the permutation. The sum of N across all test
    # cases does not exceed 2*10^5, so this will be efficient.

    out = []
    # Precompute prefix sums for sum of 1..n if needed, but here we only need
    # prefix_max and suffix_min, plus a quick identity check.

    # Process each test case
    for _ in range(t):
        N = int(input_data[idx]); idx += 1
        P = list(map(int, input_data[idx:idx+N]))
        idx += N
        
        # Check if already identity
        identity = True
        for i in range(N):
            if P[i] != i+1:
                identity = False
                break
        if identity:
            out.append("0")
            continue
        
        # Build prefix_max and suffix_min
        prefix_max = [0]*N
        suffix_min = [0]*N
        prefix_max[0] = P[0]
        for i in range(1, N):
            prefix_max[i] = max(prefix_max[i-1], P[i])
        suffix_min[N-1] = P[N-1]
        for i in range(N-2, -1, -1):
            suffix_min[i] = min(suffix_min[i+1], P[i])
        
        # Try to find a k that satisfies the condition for 1 operation
        # Condition: 
        #   - if k>1 => prefix_max[k-2] == k-1
        #   - if k<N => suffix_min[k] == k+1
        #   - P[k-1] == k
        one_op_possible = False
        for k in range(1, N+1):
            if P[k-1] != k:
                continue
            if k > 1 and prefix_max[k-2] != (k-1):
                continue
            if k < N and suffix_min[k] != (k+1):
                continue
            # We found a valid k
            one_op_possible = True
            break
        
        if one_op_possible:
            out.append("1")
        else:
            out.append("2")
    
    print("
".join(out))

def main():
    solve()

# Call solve() if needed (the problem statement requires solve() to be called).
if __name__ == "__main__":
    main()