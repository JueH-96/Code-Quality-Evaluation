def solve():
    n, q = map(int, input().split())
    edges = set()
    
    for _ in range(q):
        query = list(map(int, input().split()))
        
        if query[0] == 1:
            u, v = query[1], query[2]
            edges.add(tuple(sorted((u, v))))
        else:
            v = query[1]
            to_remove = []
            for edge in edges:
                if v in edge:
                    to_remove.append(edge)
            for edge in to_remove:
                edges.remove(edge)
        
        isolated_count = 0
        for i in range(1, n + 1):
            is_isolated = True
            for edge in edges:
                if i in edge:
                    is_isolated = False
                    break
            if is_isolated:
                isolated_count += 1
        print(isolated_count)

solve()