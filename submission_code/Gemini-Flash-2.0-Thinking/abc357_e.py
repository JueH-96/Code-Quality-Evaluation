def solve():
    n = int(input())
    a = list(map(int, input().split()))
    adj = [x - 1 for x in a]

    reachable_pairs = 0
    for start_node in range(n):
        reachable = set()
        q = [start_node]
        reachable.add(start_node)
        head = 0
        while head < len(q):
            u = q[head]
            head += 1
            v = adj[u]
            if v not in reachable:
                reachable.add(v)
                q.append(v)
        reachable_pairs += len(reachable)
    print(reachable_pairs)

solve()