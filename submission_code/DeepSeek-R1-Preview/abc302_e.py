def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    Q = int(data[idx])
    idx += 1

    adj = [set() for _ in range(N + 1)]
    degree = [0] * (N + 1)
    count = N

    for _ in range(Q):
        t = data[idx]
        idx += 1
        if t == '1':
            u = int(data[idx])
            idx += 1
            v = int(data[idx])
            idx += 1
            if v not in adj[u]:
                adj[u].add(v)
                adj[v].add(u)
                for x in [u, v]:
                    prev = degree[x]
                    degree[x] += 1
                    if prev == 0:
                        count -= 1
        else:
            v = int(data[idx])
            idx += 1
            edges_to_remove = list(adj[v])
            for u in edges_to_remove:
                adj[v].remove(u)
                adj[u].remove(v)
                prev_degree_u = degree[u]
                degree[u] -= 1
                if prev_degree_u == 1:
                    count += 1
            prev_degree_v = degree[v]
            degree[v] = 0
            if prev_degree_v > 0:
                count += 1
        print(count)

if __name__ == '__main__':
    main()