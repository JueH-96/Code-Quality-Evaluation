def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    # Read the 3x3 grid as a flat list c[0..8]
    c = list(map(int, data))
    # Predefine the 8 lines (3 rows, 3 columns, 2 diagonals)
    lines = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),
        (0, 3, 6), (1, 4, 7), (2, 5, 8),
        (0, 4, 8), (2, 4, 6)
    ]
    # Collect only those lines that have exactly two equal values
    D_lines = []
    for (p0, p1, p2) in lines:
        v0, v1, v2 = c[p0], c[p1], c[p2]
        # skip all-distinct or all-equal (all-equal won't happen by problem guarantee)
        if v0 == v1 and v1 == v2:
            continue
        if v0 == v1:
            # duplicates at p0,p1; unique at p2
            D_lines.append((p0, p1, p2))
        elif v1 == v2:
            # duplicates at p1,p2; unique at p0
            D_lines.append((p1, p2, p0))
        elif v2 == v0:
            # duplicates at p0,p2; unique at p1
            D_lines.append((p0, p2, p1))
        # else all distinct: ignore
    
    m = len(D_lines)
    # Total permutations of 9 items
    total = 362880  # 9!
    full_mask = (1 << 9) - 1  # mask for all 9 elements
    good_count = 0
    
    # We'll do inclusion-exclusion over subsets of D_lines
    # For each subset S, we build precedence constraints:
    #   for each line in S: pos[dup1] < pos[unique] and pos[dup2] < pos[unique]
    # Count f(S) = # permutations satisfying those constraints.
    # Then add sign * f(S) to good_count, sign = (-1)^|S|.
    
    # Pre-allocate dp array once
    dp = [0] * (1 << 9)
    
    for S in range(1 << m):
        # Determine sign from parity of bits in S
        if (S.bit_count() & 1) == 0:
            sign = 1
        else:
            sign = -1
        # Build preds: for each node i, preds[i] is bitmask of its required predecessors
        preds = [0] * 9
        for j in range(m):
            if (S >> j) & 1:
                a, b, u = D_lines[j]
                preds[u] |= (1 << a) | (1 << b)
        # Count linear extensions of this partial order by DP on subsets of placed nodes
        # dp[mask] = number of ways to place exactly the set "mask" so far
        # transitions: pick any i not in mask whose preds are all in mask
        for i in range(1 << 9):
            dp[i] = 0
        dp[0] = 1
        for mask in range(1 << 9):
            ways = dp[mask]
            if ways == 0:
                continue
            # try to place a new node i
            # i must not be in mask, and preds[i] must be subset of mask
            for i in range(9):
                if not (mask & (1 << i)):
                    if (preds[i] & mask) == preds[i]:
                        dp[mask | (1 << i)] += ways
        fS = dp[full_mask]
        good_count += sign * fS
    
    # Probability = good_count / total permutations
    prob = good_count / total
    # Print with sufficient precision
    print(f"{prob:.12f}")

# Call main to execute
if __name__ == "__main__":
    main()