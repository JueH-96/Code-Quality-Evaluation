import sys
sys.setrecursionlimit(1 << 25)

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    K = int(data[idx+1])
    idx +=2

    edges = [[] for _ in range(N+1)]
    for _ in range(N-1):
        a = int(data[idx])
        b = int(data[idx+1])
        edges[a].append(b)
        edges[b].append(a)
        idx +=2

    V = list(map(int, data[idx:idx+K]))
    k_set = set(V)

    s = [0] * (N + 1)
    visited = [False] * (N + 1)
    parent = [0] * (N + 1)

    def dfs(u, parent_u):
        visited[u] = True
        cnt = 0
        if u in k_set:
            cnt = 1
        for v in edges[u]:
            if not visited[v]:
                parent[v] = u
                dfs(v, u)
                cnt += s[v]
        s[u] = cnt
        return cnt

    dfs(1, -1)

    count = 0
    for u in range(1, N+1):
        if u in k_set:
            count += 1
        else:
            if s[u] > 0 and (K - s[u]) > 0:
                count += 1

    print(count)

if __name__ == '__main__':
    main()