def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))

    from collections import defaultdict

    # Dictionary: height -> list of indices where that height appears
    positions_by_height = defaultdict(list)
    for i, h in enumerate(H):
        positions_by_height[h].append(i+1)  # Store 1-based positions

    best = 1  # At least choosing one building is always valid
    for height, positions in positions_by_height.items():
        m = len(positions)
        if m <= 1:
            # Only one building of this height -> maximum length is 1
            continue

        # DP approach: dp[j] is a dictionary mapping difference -> max AP length ending at index j
        dp = [dict() for _ in range(m)]
        local_best = 1

        for j in range(m):
            for i in range(j):
                diff = positions[j] - positions[i]
                if diff in dp[i]:
                    dp[j][diff] = dp[i][diff] + 1
                else:
                    dp[j][diff] = 2  # Starting a new AP with these two
                if dp[j][diff] > local_best:
                    local_best = dp[j][diff]

        if local_best > best:
            best = local_best

    print(best)

# Do not forget to call main()!
if __name__ == "__main__":
    main()