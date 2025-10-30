import sys
input = sys.stdin.read

def max_weight_matching(N, D):
    # Initialize a memoization table
    memo = [[-1] * (1 << N) for _ in range(N)]

    def dp(last, mask):
        if mask == (1 << N) - 1:
            return 0
        if memo[last][mask] != -1:
            return memo[last][mask]

        max_weight = 0
        for next in range(N):
            if (mask & (1 << next)) == 0 and (last == -1 or D[last][next] != -1):
                weight = D[last][next] if last != -1 else 0
                weight += dp(next, mask | (1 << next))
                max_weight = max(max_weight, weight)

        memo[last][mask] = max_weight
        return max_weight

    # Convert the input list to a 2D list for easier access
    graph = [[-1] * N for _ in range(N)]
    index = 0
    for i in range(N):
        for j in range(i + 1, N):
            graph[i][j] = D[index]
            graph[j][i] = D[index]
            index += 1

    result = dp(-1, 0)
    return result

# Read input
data = input().strip().split()
N = int(data[0])
D = list(map(int, data[1:]))

# Calculate and print the result
print(max_weight_matching(N, D))