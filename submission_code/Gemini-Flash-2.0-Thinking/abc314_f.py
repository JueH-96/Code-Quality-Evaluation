def solve():
    n = int(input())
    matches = []
    for _ in range(n - 1):
        matches.append(list(map(int, input().split())))

    mod = 998244353

    parent = list(range(n + 1))
    size = [1] * (n + 1)
    expected_wins_team = [0] * (n + 1)

    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            if size[root_i] < size[root_j]:
                root_i, root_j = root_j, root_i
            parent[root_j] = root_i
            size[root_i] += size[root_j]

    for p, q in matches:
        root_p = find(p)
        root_q = find(q)

        if root_p != root_q:
            size_p = size[root_p]
            size_q = size[root_q]

            prob_p_wins = (size_p * pow(size_p + size_q, mod - 2, mod)) % mod
            prob_q_wins = (size_q * pow(size_p + size_q, mod - 2, mod)) % mod

            expected_wins_team[root_p] = (expected_wins_team[root_p] + prob_p_wins) % mod
            expected_wins_team[root_q] = (expected_wins_team[root_q] + prob_q_wins) % mod

            union(p, q)

    result = [0] * n
    team_of_player = [0] * (n + 1)
    for i in range(1, n + 1):
        team_of_player[i] = find(i)

    team_expected_wins = {}
    for i in range(1, n + 1):
        root = find(i)
        if root not in team_expected_wins:
            team_expected_wins[root] = 0

    for k in range(n - 1):
        p, q = matches[k]
        root_p = find(p)
        root_q = find(q)

        if root_p != root_q:
            size_p = size[root_p]
            size_q = size[root_q]
            prob_p_wins = (size_p * pow(size_p + size_q, mod - 2, mod)) % mod
            prob_q_wins = (size_q * pow(size_p + size_q, mod - 2, mod)) % mod
            team_expected_wins[root_p] = (team_expected_wins[root_p] + prob_p_wins) % mod
            team_expected_wins[root_q] = (team_expected_wins[root_q] + prob_q_wins) % mod
            union(p, q)

    final_expected_wins = [0] * n
    for i in range(1, n + 1):
        final_expected_wins[i-1] = team_expected_wins[find(i)]

    print(*final_expected_wins)

solve()