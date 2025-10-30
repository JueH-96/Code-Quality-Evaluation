def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    D = list(map(int, input[idx:idx+N]))
    idx += N
    L1 = int(input[idx])
    C1 = int(input[idx+1])
    K1 = int(input[idx+2])
    idx += 3
    L2 = int(input[idx])
    C2 = int(input[idx+1])
    K2 = int(input[idx+2])
    idx += 3

    INF = float('inf')

    # Initialize DP table
    dp = [ [INF] * (K2 + 1) for _ in range(K1 + 1) ]
    dp[0][0] = 0

    for d in D:
        possible_pairs = []
        for x in range(0, K1 + 1):
            if x * L1 >= d:
                y = 0
            else:
                remaining = d - x * L1
                y = (remaining + L2 - 1) // L2
            if y < 0:
                y = 0
            possible_pairs.append((x, y))
        
        # Remove duplicate pairs
        unique_pairs = list(dict.fromkeys(possible_pairs))
        possible_pairs = unique_pairs

        # Create a new DP table for this section
        new_dp = [ [INF] * (K2 + 1) for _ in range(K1 + 1) ]

        for a in range(K1 + 1):
            for b in range(K2 + 1):
                if dp[a][b] == INF:
                    continue
                for x, y in possible_pairs:
                    new_a = a + x
                    new_b = b + y
                    if new_a > K1 or new_b > K2:
                        continue
                    new_cost = dp[a][b] + x * C1 + y * C2
                    if new_cost < new_dp[new_a][new_b]:
                        new_dp[new_a][new_b] = new_cost

        # Update dp for the next iteration
        dp = new_dp

    # Find the minimal cost after all sections
    min_cost = INF
    for a in range(K1 + 1):
        for b in range(K2 + 1):
            if dp[a][b] < min_cost:
                min_cost = dp[a][b]

    if min_cost == INF:
        print(-1)
    else:
        print(min_cost)

if __name__ == '__main__':
    main()