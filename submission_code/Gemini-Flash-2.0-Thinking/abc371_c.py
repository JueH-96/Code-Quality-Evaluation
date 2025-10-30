import itertools

def solve():
    n = int(input())
    mg = int(input())
    adj_g = [[False] * n for _ in range(n)]
    for _ in range(mg):
        u, v = map(int, input().split())
        adj_g[u - 1][v - 1] = adj_g[v - 1][u - 1] = True

    mh = int(input())
    adj_h = [[False] * n for _ in range(n)]
    for _ in range(mh):
        u, v = map(int, input().split())
        adj_h[u - 1][v - 1] = adj_h[v - 1][u - 1] = True

    costs = []
    for _ in range(n - 1):
        costs.append(list(map(int, input().split())))

    def get_cost(u, v):
        if u > v:
            u, v = v, u
        return costs[u - 1][v - u - 1]

    min_total_cost = float('inf')

    for perm in itertools.permutations(range(n)):
        current_cost = 0
        for i in range(n):
            for j in range(i + 1, n):
                g_edge = adj_g[i][j]
                h_edge = adj_h[perm[i]][perm[j]]

                if g_edge and not h_edge:
                    current_cost += get_cost(perm[i] + 1, perm[j] + 1)
                elif not g_edge and h_edge:
                    current_cost += get_cost(perm[i] + 1, perm[j] + 1)

        min_total_cost = min(min_total_cost, current_cost)

    print(min_total_cost)

if __name__ == "__main__":
    solve()