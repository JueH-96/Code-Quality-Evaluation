def solve():
    n, m = map(int, input().split())
    adj = [[] for _ in range(n)]
    for _ in range(m):
        u, v = map(int, input().split())
        adj[u - 1].append(v - 1)

    def can_reach(start, end, visited):
        visited[start] = True
        if start == end:
            return True
        for neighbor in adj[start]:
            if not visited[neighbor]:
                if can_reach(neighbor, end, visited):
                    return True
        return False

    possible_strongest = []
    for i in range(n):
        is_strongest = True
        for j in range(n):
            if i == j:
                continue
            visited = [False] * n
            if not can_reach(i, j, visited):
                is_strongest = False
                break
        if is_strongest:
            possible_strongest.append(i + 1)

    if len(possible_strongest) == 1:
        print(possible_strongest[0])
    else:
        print(-1)

solve()