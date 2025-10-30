def main():
    import sys
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    
    # For a given contiguous subarray [L, R], let D be the set of distinct numbers.
    # Our operations allow us to erase D in (# of maximal contiguous intervals in D) moves.
    # Notice that if D has d distinct numbers and if there are c “adjacent” pairs (x, x+1)
    # present in D, then f(L,R) = d - c.
    # Hence, summing f(L,R) over all subarrays is equivalent to:
    #   (sum over subarrays of [# distinct numbers]) - (sum over subarrays of [# adjacent consecutive pairs]).
    # We now explain a plan for computing each part.
    #
    # Let T1 = sum_{L,R} (# distinct numbers in subarray).
    # For each value v (1 <= v <= N), if we let I(v) be the indicator that v appears in a subarray,
    # then the total contribution of v is (# of subarrays containing v).
    # We can compute that if we know the list of positions where v occurs.
    #
    # Similarly, let T2 = sum_{L,R} (# of pairs (v, v+1) that occur in the subarray).
    # For each consecutive pair (v, v+1), if both occur in the subarray then they contribute 1.
    # So T2 is the sum, over all v, of (# subarrays that contain both v and v+1).
    #
    # The final answer is T1 - T2.
    #
    # We now describe how to compute the number of subarrays that contain a given value.
    # Let total subarrays = tot_sub = N*(N+1)//2.
    # For a given sorted list of positions P (with 1-indexed positions), the subarrays
    # that do NOT contain v are exactly those that lie entirely in one of the “gaps”
    # between occurrences. With boundaries 0 and N+1, if the gaps are:
    #   gap0 = P[0] - 1, gap1 = P[1] - P[0] - 1, ..., gap_last = N - P[-1],
    # then the number of subarrays that avoid v is:
    #   sum_{each gap} gap*(gap+1)//2.
    # Thus, # subarrays with v = tot_sub - (sum of gap counts).
    
    # Build position arrays (1-indexed) for each value in 1..N.
    pos = [[] for _ in range(N+1)]
    for i, a in enumerate(A, start=1):
        pos[a].append(i)
    
    tot_sub = N * (N + 1) // 2

    # Helper function: given a sorted list of positions P,
    # compute the number of subarrays that do NOT contain any occurrence from P.
    def missing_count(P, N):
        last = 0
        tot = 0
        for x in P:
            gap = x - last - 1
            tot += gap * (gap + 1) // 2
            last = x
        gap = N - last
        tot += gap * (gap + 1) // 2
        return tot

    # For each v, if it appears, compute:
    #   missing_arr[v] = number of subarrays that do NOT contain v,
    #   contain_arr[v] = tot_sub - missing_arr[v] (i.e. subarrays that do contain v).
    missing_arr = [tot_sub] * (N + 1)  # if v never appears, missing_arr = tot_sub so contain count = 0.
    contain_arr = [0] * (N + 1)
    T1 = 0
    for v in range(1, N+1):
        if pos[v]:
            mcnt = missing_count(pos[v], N)
            missing_arr[v] = mcnt
            c = tot_sub - mcnt
            contain_arr[v] = c
            T1 += c

    # Now T1 is the sum, over all subarrays, of (# distinct numbers in that subarray).

    # Next, for each adjacent pair (v, v+1) we want to count the number of subarrays
    # that contain both v and v+1. In a subarray S, the indicator for (v, v+1) being present
    # is 1 if and only if v and v+1 occur.
    # By inclusion-exclusion, for each such pair:
    #    count(v and v+1) = tot_sub - missing(v) - missing(v+1) + missing_both,
    # where missing(v) and missing(v+1) use the precomputed missing_arr,
    # and missing_both is computed from the merged sorted list U = merge(pos[v], pos[v+1]).
    
    # Helper function: merge two sorted lists.
    def merge_two(P, Q):
        i, j = 0, 0
        merged = []
        while i < len(P) and j < len(Q):
            if P[i] < Q[j]:
                merged.append(P[i])
                i += 1
            else:
                merged.append(Q[j])
                j += 1
        if i < len(P):
            merged.extend(P[i:])
        if j < len(Q):
            merged.extend(Q[j:])
        return merged

    T2 = 0
    for v in range(1, N):
        # Only if both v and v+1 appear at least once.
        if pos[v] and pos[v+1]:
            union_list = merge_two(pos[v], pos[v+1])
            miss_union = missing_count(union_list, N)
            cnt_both = tot_sub - missing_arr[v] - missing_arr[v+1] + miss_union
            T2 += cnt_both

    # The answer is the sum over subarrays f(L, R) = (# distinct in S) - (# adjacent pairs in S)
    ans = T1 - T2
    sys.stdout.write(str(ans))

if __name__ == '__main__':
    main()