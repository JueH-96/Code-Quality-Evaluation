def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    A = list(map(int, data[1:]))

    # Collect the indices for each value. We store 1-indexed positions.
    positions = {}
    for idx, x in enumerate(A):
        if x not in positions:
            positions[x] = []
        positions[x].append(idx + 1)
    
    # For a group with m occurrences at sorted positions p[0], p[1], ... p[m-1]
    # we want to sum for each pair (i, k) with i < k:
    #    ( (p[k] - p[i] - 1) - (# of indices between that are equal to this value) )
    # Notice that if the positions list is [p0, p1, ..., p[m-1]], then for any pair
    # (p[r], p[s]) with r < s the number of positions between is (p[s]-p[r]-1)
    # and # of occurrences of the value between is exactly (s - r - 1).
    # Thus the contribution of that pair is: (p[s]-p[r]) - (s - r).
    #
    # We can sum this over all pairs for a given value:
    #     f = sum_{r < s} (p[s] - p[r]) - sum_{r < s} (s - r)
    #
    # The first term can be computed using a known trick:
    #   sum_{i=0}^{m-1} p[i]*(2*i - m + 1)
    #
    # The second term:
    #   sum_{d=1}^{m-1} d*(m-d)
    # can be simplified into a closed form:
    #   S = m*(m-1)*(m+1)//6
    #
    # Our answer is the sum of f for all values that occur at least twice.
    
    total = 0
    for x, pos_list in positions.items():
        m = len(pos_list)
        if m < 2:
            continue
        group_sum = 0
        # The positions are collected in increasing order.
        # Compute sum_{i=0}^{m-1} pos_list[i]*(2*i - m + 1)
        for i, pos in enumerate(pos_list):
            group_sum += pos * (2 * i - m + 1)
        # Compute the constant second summation:
        S = m * (m - 1) * (m + 1) // 6
        total += group_sum - S
    
    sys.stdout.write(str(total))


if __name__ == '__main__':
    main()