import sys

def solve():
    """
    Solves a single test case of the Empty Sequence problem.
    """
    try:
        line = sys.stdin.readline()
        if not line: return
        N = int(line)
        A = list(map(int, sys.stdin.readline().split()))
    except (IOError, ValueError):
        return

    if N == 0:
        print(0)
        return

    distinct_vals = sorted(list(set(A)))
    D = len(distinct_vals)

    if D == 1:
        print(1)
        return
    
    if D == N:
        print(N)
        return

    val_to_id = {val: i for i, val in enumerate(distinct_vals)}
    pos = [[] for _ in range(D)]
    for i in range(N):
        val_id = val_to_id[A[i]]
        pos[val_id].append(i)

    # Strategy 1: DP on counts (for D=2, where it's feasible)
    # This correctly handles cases like Sample 3.
    if D == 2:
        c0, c1 = len(pos[0]), len(pos[1])
        dp = [[float('inf')] * (c1 + 1) for _ in range(c0 + 1)]
        dp[0][0] = 0

        cost0_prefix_sum = [[0] * (c0 + 1) for _ in range(c1 + 1)]
        for j in range(c1 + 1):
            p1_ptr = 0
            for i in range(1, c0 + 1):
                while p1_ptr < j and pos[1][p1_ptr] < pos[0][i - 1]:
                    p1_ptr += 1
                displacement = pos[0][i - 1] - (i - 1) - p1_ptr
                cost0_prefix_sum[j][i] = cost0_prefix_sum[j][i - 1] + displacement

        cost1_prefix_sum = [[0] * (c1 + 1) for _ in range(c0 + 1)]
        for i in range(c0 + 1):
            p0_ptr = 0
            for j in range(1, c1 + 1):
                while p0_ptr < i and pos[0][p0_ptr] < pos[1][j - 1]:
                    p0_ptr += 1
                displacement = pos[1][j - 1] - (j - 1) - p0_ptr
                cost1_prefix_sum[i][j] = cost1_prefix_sum[i][j - 1] + displacement

        for i in range(c0 + 1):
            for j in range(c1 + 1):
                if i > 0:
                    for k in range(1, i + 1):
                        swaps = cost0_prefix_sum[j][i] - cost0_prefix_sum[j][i - k]
                        dp[i][j] = min(dp[i][j], dp[i - k][j] + 1 + swaps)
                if j > 0:
                    for l in range(1, j + 1):
                        swaps = cost1_prefix_sum[i][j] - cost1_prefix_sum[i][j - l]
                        dp[i][j] = min(dp[i][j], dp[i][j - l] + 1 + swaps)
        print(dp[c0][c1])
        return

    # Strategy 2: DP on subsets of values (for D > 2)
    # This assumes a simpler group-and-delete model.
    inv = [[0] * D for _ in range(D)]
    for i in range(D):
        for j in range(i + 1, D):
            inv_ji = 0
            p1, p2 = 0, 0
            pos_i, pos_j = pos[i], pos[j]
            len_i, len_j = len(pos_i), len(pos_j)
            while p2 < len_j:
                while p1 < len_i and pos_i[p1] < pos_j[p2]:
                    p1 += 1
                inv_ji += p1
                p2 += 1
            inv[j][i] = inv_ji
            inv[i][j] = len_i * len_j - inv_ji
    
    dp = [float('inf')] * (1 << D)
    dp[0] = 0
    for mask in range(1, 1 << D):
        for i in range(D):
            if (mask >> i) & 1:
                prev_mask = mask ^ (1 << i)
                swaps = 0
                for j in range(D):
                    if not ((mask >> j) & 1):
                        swaps += inv[j][i]
                cost = dp[prev_mask] + 1 + swaps
                dp[mask] = min(dp[mask], cost)
    print(dp[(1 << D) - 1])

def main():
    try:
        T_str = sys.stdin.readline()
        if not T_str: return
        T = int(T_str)
        for _ in range(T):
            solve()
    except (IOError, ValueError):
        return

if __name__ == "__main__":
    main()