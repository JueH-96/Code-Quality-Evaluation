def solve():
    n = int(input())
    weights_input = []
    for _ in range(n - 1):
        weights_input.append(list(map(int, input().split())))

    weights = {}
    for i in range(n - 1):
        for j in range(i + 1, n):
            weights[(i, j)] = weights_input[i][j - i - 1]

    memo = {}

    def dp(mask):
        if mask == 0:
            return 0
        if mask in memo:
            return memo[mask]

        vertices = [i for i in range(n) if (mask >> i) & 1]
        num_vertices = len(vertices)

        if num_vertices % 2 != 0:
            memo[mask] = 0
            return 0

        max_weight = 0
        for i_idx in range(num_vertices):
            for j_idx in range(i_idx + 1, num_vertices):
                u = vertices[i_idx]
                v = vertices[j_idx]
                weight = weights.get((min(u, v), max(u, v)), 0)
                remaining_mask = mask ^ (1 << u) ^ (1 << v)
                max_weight = max(max_weight, dp(remaining_mask) + weight)

        memo[mask] = max_weight
        return max_weight

    initial_mask = (1 << n) - 1
    result = dp(initial_mask)
    print(result)

solve()