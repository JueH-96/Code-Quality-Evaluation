def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    it = iter(input_data)
    n = int(next(it))
    M = int(next(it))
    A = [int(next(it)) for _ in range(n)]

    # We will use a Fenwick Tree (Binary Indexed Tree) to maintain:
    # 1. The count of previous prefix remainders at each value.
    # 2. The sum of the actual remainders.
    #
    # Let P[k] be the prefix sums mod M with P[0]=0.
    # Then, for any subarray from l to r (with l>=1) the remainder is:
    #   R = (P[r] - P[l-1]) mod M,
    # which equals:
    #   if P[r] >= P[l-1]: (P[r]-P[l-1])
    #   else: (P[r]-P[l-1] + M)
    #
    # Fixing r and letting x = P[r], consider all previous indices i (0 <= i < r) with remainder y = P[i]:
    #   - If y <= x, contribution = x - y.
    #   - If y > x, contribution = x - y + M.
    #
    # For each new prefix remainder, we can use a Fenwick tree to quickly query:
    #   count_lower, sum_lower for indices with remainder <= x.
    # And similarly, using total counts we can compute the count and sum for the "higher" group.
    
    # Fenwick tree size is M (possible remainders: 0, 1, ..., M-1)
    size = M
    # Using 1-indexed Fenwick trees:
    fenwCount = [0] * (size + 1)  # frequency tree
    fenwSum = [0] * (size + 1)    # sum tree
    
    # Fenwick tree update: update position i (0-indexed value) with count delta and sum delta.
    def fenw_update(i, delta_count, delta_sum):
        idx = i + 1  # shifting to 1-indexed
        while idx <= size:
            fenwCount[idx] += delta_count
            fenwSum[idx] += delta_sum
            idx += idx & -idx

    # Fenwick tree query: gets (count, sum) for indices [0, i] (0-indexed).
    def fenw_query(i):
        total_count, total_sum = 0, 0
        idx = i + 1
        while idx:
            total_count += fenwCount[idx]
            total_sum += fenwSum[idx]
            idx -= idx & -idx
        return total_count, total_sum

    # Initially, we consider the prefix sum P[0] = 0.
    fenw_update(0, 1, 0)

    total_answer = 0
    curr_prefix = 0
    # Process each element to update prefix remainder and compute contribution.
    for a in A:
        curr_prefix = (curr_prefix + a) % M   # new prefix remainder P[i]
        # Query the fenwick tree for remainders <= curr_prefix.
        count_lower, sum_lower = fenw_query(curr_prefix)
        # Total count and sum up to remainder M-1 (i.e., for all previous remainders)
        total_count, total_sum = fenw_query(M - 1)
        count_higher = total_count - count_lower
        sum_higher = total_sum - sum_lower
        # For previous remainders:
        #  Group1 (y <= curr_prefix): sum of contributions = curr_prefix*count_lower - sum_lower.
        #  Group2 (y > curr_prefix): sum = curr_prefix*count_higher - sum_higher + M * count_higher.
        contribution = (curr_prefix * count_lower - sum_lower) + \
                       (curr_prefix * count_higher - sum_higher + M * count_higher)
        total_answer += contribution
        # Insert the current prefix remainder into the Fenwick tree.
        fenw_update(curr_prefix, 1, curr_prefix)
    
    sys.stdout.write(str(total_answer))
    
if __name__ == "__main__":
    main()