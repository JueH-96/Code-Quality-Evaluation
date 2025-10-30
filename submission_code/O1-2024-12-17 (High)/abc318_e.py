def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    from collections import defaultdict
    # Dictionary mapping A_i values to a list of positions where they appear
    positions_by_value = defaultdict(list)

    # Collect positions (1-based) for each distinct value
    for i, val in enumerate(A):
        positions_by_value[val].append(i + 1)

    answer = 0

    # For each distinct value, calculate contribution to the result
    for val, pos_list in positions_by_value.items():
        m = len(pos_list)
        if m < 2:
            continue  # No pairs if there's only one position

        # We'll use prefix sums to compute the desired sum of:
        #   Σ ( (p[y] - p[x]) - ( (y+1) - (x+1) ) ) for 0 <= x < y < m
        # which can be transformed into:
        #   Σ ( Q[y] - Q[x] ), Q[i] = p[i] - i (0-based for i)
        # The efficient form is:
        #   Σ_{y=1..m-1} [ y * Q[y] - Σ_{x=0..y-1} Q[x] ].

        prefix_sum = [0] * (m + 1)  # prefix_sum[i] = sum of Q[x] for x in [0..i-1]
        for i in range(m):
            prefix_sum[i + 1] = prefix_sum[i] + (pos_list[i] - i)

        # Calculate the contributions
        for y in range(1, m):
            Qy = pos_list[y] - y
            answer += y * Qy - prefix_sum[y]

    print(answer)

# Do not forget to call main()
main()