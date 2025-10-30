def main():
    import sys

    # Read input
    input = sys.stdin.read().splitlines()
    N = int(input[0])
    weight = [[0] * N for _ in range(N)]
    index = 1
    for i in range(N-1):
        line = list(map(int, input[index].split()))
        for j in range(i+1, N):
            weight[i][j] = line[j-i-1]
            weight[j][i] = weight[i][j]
        index += 1

    # Initialize dp array
    dp = [0] * (1 << N)

    # Iterate over all subsets
    for mask in range(1, 1 << N):
        best = 0
        # Get the list of vertices in the current subset
        vertices = [i for i in range(N) if mask & (1 << i)]
        # Iterate over all pairs in the subset
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                u = vertices[i]
                v = vertices[j]
                new_mask = mask ^ (1 << u) ^ (1 << v)
                if new_mask < mask:
                    best = max(best, dp[new_mask] + weight[u][v])
        dp[mask] = best

    # Output the result for the full set
    print(dp[(1 << N) - 1])

if __name__ == "__main__":
    main()