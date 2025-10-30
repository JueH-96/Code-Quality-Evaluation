def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    A = data[1].strip()
    # Number of leaves = 3^N
    n_leaves = 3**N

    # We'll set up a bottom‐up DP for a full 3-ary tree.
    # For every segment (at a leaf or an internal node) we want to compute a tuple:
    #   (min_changes_to_force_value_0, min_changes_to_force_value_1)
    #
    # At the leaves, the cost is 0 if the leaf already has the target bit, or 1 if we need to flip it.
    dp = []
    for ch in A:
        # dp for leaf is:
        #   cost[0] = 0 if ch is '0', else 1;
        #   cost[1] = 0 if ch is '1', else 1.
        dp.append((0 if ch == '0' else 1, 0 if ch == '1' else 1))
    
    # For an internal node, its value is defined by taking groups of three children and using their majority.
    # To force a parent's value v (either 0 or 1) we need to assign values to each of its three children
    # such that the majority is v. In a group of three, the possible assignments (as bits) that yield a majority of v are:
    # For v = 0: (0,0,0), (0,0,1), (0,1,0), (1,0,0)
    # For v = 1: (1,1,1), (1,1,0), (1,0,1), (0,1,1)
    valid_assignments = {0: [], 1: []}
    for a in [0, 1]:
        for b in [0, 1]:
            for c in [0, 1]:
                # Majority is 0 if at least 2 zeros:
                if (a + b + c) <= 1:
                    valid_assignments[0].append((a, b, c))
                # Majority is 1 if at least 2 ones:
                if (a + b + c) >= 2:
                    valid_assignments[1].append((a, b, c))
    
    # Given the DP results for three children (each as (cost_to_0, cost_to_1)),
    # the parent's cost to be forced to v is the minimal possible sum of the three children’s change costs
    # where the choices (assignment bits for the three children) yield a majority of v.
    def combine(dp1, dp2, dp3):
        res0 = float('inf')
        res1 = float('inf')
        for v in [0, 1]:
            best = float('inf')
            for assign in valid_assignments[v]:
                cost = dp1[assign[0]] + dp2[assign[1]] + dp3[assign[2]]
                if cost < best:
                    best = cost
            if v == 0:
                res0 = best
            else:
                res1 = best
        return (res0, res1)
    
    # Build the DP tree bottom-up. At each level, group nodes in groups of 3.
    while len(dp) > 1:
        new_dp = []
        for i in range(0, len(dp), 3):
            new_dp.append(combine(dp[i], dp[i+1], dp[i+2]))
        dp = new_dp
    
    # At this point dp[0] is a tuple (cost_to_force_0, cost_to_force_1) for the top.
    # However, the problem asks for the minimum number of changes required in A to flip the final outcome.
    
    # First, simulate the original process (with zero changes) to obtain the final bit.
    current = A
    for _ in range(N):
        next_level = []
        for i in range(0, len(current), 3):
            group = current[i:i+3]
            # The majority of group: if there are at least 2 zeros, value is 0; otherwise, it is 1.
            if group.count('0') >= 2:
                next_level.append('0')
            else:
                next_level.append('1')
        current = "".join(next_level)
    original_final = current  # This is a string of length 1.
    # We want to flip the final value.
    desired = '1' if original_final == '0' else '0'
    desired_val = 0 if desired == '0' else 1
    ans = dp[0][desired_val]
    sys.stdout.write(str(ans))


if __name__ == '__main__':
    main()