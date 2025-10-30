def solve():
    n, q = map(int, input().split())
    edges = {}
    
    def count_isolated():
        count = 0
        for i in range(1, n + 1):
            is_isolated = True
            for u, v in edges.items():
                if i in u or i in v:
                    is_isolated = False
                    break
            if is_isolated:
                count += 1
        return count
    
    for _ in range(q):
        query = list(map(int, input().split()))
        
        if query[0] == 1:
            u, v = query[1], query[2]
            edges[(min(u,v), max(u,v))] = True
        elif query[0] == 2:
            v = query[1]
            to_remove = []
            for edge in edges:
                if v in edge:
                    to_remove.append(edge)
            for edge in to_remove:
                del edges[edge]
        
        print(count_isolated())

solve()