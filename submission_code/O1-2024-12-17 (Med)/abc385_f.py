def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    X = []
    H = []
    ptr = 1
    for _ in range(N):
        x_i = int(input_data[ptr]); h_i = int(input_data[ptr+1])
        ptr += 2
        X.append(x_i)
        H.append(h_i)

    # --------------------------------------------------------------------------------
    # We wish to find the largest nonnegative h such that from (0,h) we do NOT see
    # at least one building. Equivalently, we find the "critical" h above which
    # all buildings become visible, and that boundary is the answer.
    #
    # From earlier reasoning (and verified against the samples), a succinct way
    # to compute that boundary is via:
    #
    #   M_i = max over j < i of  [ (X_i * H_j - X_j * H_i) / (X_i - X_j) ]
    #
    # (If there is no j < i, M_i = -∞ so building i is unblocked by any to its left.)
    #
    # Then if we let M = max_i(M_i), we can see all buildings only if h > M.
    # Hence the "maximum h that fails to see all" is M.
    # If M <= 0, then even at h=0 we already see all buildings, so answer is -1.
    #
    # The only challenge is to compute these M_i in O(N) or O(N log N) time,
    # because N can be up to 2e5.
    #
    # A known trick (sometimes called a "convex-hull"-like stack method), works
    # in a single pass from left to right (since X is strictly increasing).
    #
    # Define a function overshadow(j, i) = (X_i * H_j - X_j * H_i)/(X_i - X_j).
    # We keep a stack "st" of indices of buildings (in strictly increasing order of X).
    # Each time we consider a new building i, we compute the intersection with the
    # top of the stack, pop while that intersection is not strictly greater than the
    # previous intersection, etc.  This is analogous to the usual "convex hull trick"
    # insertion when lines have strictly decreasing slopes (here slope = -1/X, which
    # is indeed decreasing in X if X is increasing).
    #
    # We record all intersection values in an array "inter", where inter[k] is
    # the vantage-height at which the building st[k+1] first overshadows st[k].
    # The answer is then the maximum of these intersection values.
    #
    # Details:
    #  - overshadow(j, i) can be negative; that simply means for vantage h> that value,
    #    building i is on top.  We only care about the maximum such values that block
    #    visibility.  
    #  - In the end, max(inter) is our M.  If M <= 0, output -1, else M.
    #
    # Let's implement it now.
    # --------------------------------------------------------------------------------

    def overshadow(j, i):
        # Returns the vantage height at which building j first
        # blocks (or equals) building i (or vice-versa) when viewed from (0,h).
        # Formula:
        #   (X_i * H_j - X_j * H_i) / (X_i - X_j),   with X_j < X_i
        return (X[i]*H[j] - X[j]*H[i])/(X[i] - X[j])

    st = []         # stack of indices
    inter = []      # inter[k] = vantage h at which st[k+1] overshadows st[k]
                    # (we store them in parallel with st, so inter has len(st)-1 effectively)

    max_val = float('-inf')

    for i in range(N):
        # Pop while the new intersection is <= the last intersection
        # This is the usual "mono stack" pattern for lines with decreasing slopes
        # But we must have at least 2 in the stack to compare.
        while len(st) >= 2:
            meet_new = overshadow(st[-1], i)
            if meet_new <= inter[-1]:
                # the top line is never going to yield a bigger intersection
                st.pop()
                inter.pop()
            else:
                break
        
        if st:
            meet = overshadow(st[-1], i)
            inter.append(meet)
            if meet > max_val:
                max_val = meet
        else:
            # No intersection if stack empty
            inter.append(float('-inf'))
        
        st.append(i)

    # max_val is the maximum intersection
    if max_val <= 0:
        print(-1)
    else:
        # Print with (up to) 18 decimal places
        print(f"{max_val:.18f}")