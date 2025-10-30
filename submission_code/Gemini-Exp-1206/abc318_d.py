def solve():
    n = int(input())
    d = {}
    k = 0
    for i in range(1, n):
        line = list(map(int, input().split()))
        for j in range(i + 1, n + 1):
            d[(i, j)] = line[k]
            k += 1
        k = 0

    ans = 0
    for i in range(1 << (n * (n - 1) // 2)):
        edges = []
        cur = 0
        k = 0
        for r in range(1, n):
            for c in range(r + 1, n + 1):
                if (i >> k) & 1:
                    edges.append((r, c))
                k += 1

        
        nodes = set()
        weight = 0
        valid = True
        for u, v in edges:
            if u in nodes or v in nodes:
                valid = False
                break
            nodes.add(u)
            nodes.add(v)
            weight += d[(u, v)]
        
        if valid:
            ans = max(ans, weight)

    print(ans)

solve()