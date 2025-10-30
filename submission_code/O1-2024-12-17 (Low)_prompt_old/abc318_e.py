def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # positions[x] will hold all 1-based positions i where A_i = x
    positions = [[] for _ in range(N + 1)]
    for i, v in enumerate(A, start=1):
        positions[v].append(i)

    count_same_ends = 0  # Count of (i, j, k) with A_i = A_k (no condition on A_j)
    count_all_equal = 0  # Count of (i, j, k) with A_i = A_j = A_k

    # We'll iterate over each distinct value x, gather its positions,
    # then compute:
    # 1) the number of i<j<k with A_i = A_k = x
    # 2) the total number of i<j<k with A_i = A_k = x, ignoring A_j
    for x_positions in positions:
        m = len(x_positions)
        if m < 2:
            continue

        # Compute the number of i<j<k with A_i=A_k=x
        # This is sum_{1 <= a < b <= m} ( x_positions[b] - x_positions[a] - 1 )
        # We'll do this in O(m) with prefix sums of positions.
        prefix_sum = [0] * (m + 1)
        for i in range(m):
            prefix_sum[i+1] = prefix_sum[i] + x_positions[i]

        local_count = 0
        # b from 1..m-1 (indexing x_positions in 0-based)
        # for each b, the number contributed is:
        # (b) * x_positions[b] - sum_{a=0..b-1} x_positions[a] - b
        # because for each a < b (which is b choices),
        # the difference is x_positions[b]-x_positions[a]-1
        for b in range(1, m):
            local_count += b * x_positions[b] - prefix_sum[b] - b

        count_same_ends += local_count

        # Compute the number of i<j<k with A_i=A_j=A_k=x
        # i.e. choose(m, 3) among positions
        if m >= 3:
            count_all_equal += m*(m-1)*(m-2)//6

    # Now, the total number of (i, j, k) with A_i=A_k and A_i!=A_j
    # is count_same_ends - count_all_equal.
    print(count_same_ends - count_all_equal)

def main():
    solve()

# Calling solve() as per requirement
if __name__ == "__main__":
    main()