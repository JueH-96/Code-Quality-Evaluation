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
    isolated_count = N

    for _ in range(Q):
        query_type = data[idx]
        idx += 1
        if query_type == '1':
            u = int(data[idx])
            idx += 1
            v = int(data[idx])
            idx += 1
            old_u = len(adj[u])
            old_v = len(adj[v])
            adj[u].add(v)
            adj[v].add(u)
            if old_u == 0:
                isolated_count -= 1
            if old_v == 0:
                isolated_count -= 1
        else:
            v = int(data[idx])
            idx += 1
            old_degree_v = len(adj[v])
            if old_degree_v > 0:
                neighbors = list(adj[v])
                for w in neighbors:
                    old_degree_w_before = len(adj[w])
                    adj[w].remove(v)
                    adj[v].remove(w)
                    if old_degree_w_before == 1:
                        isolated_count += 1
                isolated_count += 1  # for vertex v itself
        print(isolated_count)

if __name__ == '__main__':
    main()