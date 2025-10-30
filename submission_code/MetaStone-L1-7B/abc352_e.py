import sys
sys.setrecursionlimit(1 << 25)

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    M = int(data[idx])
    idx += 1

    groups = []
    for _ in range(M):
        K_i = int(data[idx])
        idx += 1
        A_i = list(map(int, data[idx:idx + K_i]))
        idx += K_i
        groups.append((K_i, A_i))

    groups.sort(key=lambda x: x[0])

    parent = list(range(N + 1))
    rank = [1] * (N + 1)

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    edges_added = 0
    total_weight = 0

    for group in groups:
        C_i = group[0]
        S_i = group[1]
        if not S_i:
            continue
        representative = S_i[0]
        count = 0
        for u in S_i:
            if find(u) != find(representative):
                count += 1
        edges_added += (count - 1)
        total_weight += (count - 1) * C_i
        for u in S_i:
            if find(u) != find(representative):
                if rank[u] >= rank[representative]:
                    parent[representative] = u
                else:
                    parent[u] = representative
                    if rank[u] == rank[representative]:
                        rank[representative] += 1
        if edges_added == N - 1:
            break

    if edges_added == N - 1:
        print(total_weight)
    else:
        print(-1)

if __name__ == '__main__':
    main()