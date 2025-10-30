import collections

def solve():
    n, m = map(int, input().split())
    informations = []
    for _ in range(m):
        u, v, x, y = map(int, input().split())
        informations.append(((u, v), (x, y)))
    
    coords = [None] * (n + 1)
    coords[1] = (0, 0)
    
    adj = collections.defaultdict(list)
    rev_adj = collections.defaultdict(list)
    
    for (u, v), (x, y) in informations:
        adj[u].append((v, (x, y)))
        rev_adj[v].append((u, (x, y)))
        
    queue = collections.deque([1])
    determined = {1}
    
    while queue:
        u = queue.popleft()
        current_coord = coords[u]
        
        for v, (x_diff, y_diff) in adj[u]:
            if coords[v] is None:
                coords[v] = (current_coord[0] + x_diff, current_coord[1] + y_diff)
                queue.append(v)
                determined.add(v)
                
        for u_rev, (x_diff, y_diff) in rev_adj[u]:
            v_rev = u_rev
            x_rev_diff, y_rev_diff = x_diff, y_diff
            if coords[v_rev] is None:
                coords[v_rev] = (current_coord[0] - x_rev_diff, current_coord[1] - y_rev_diff)
                queue.append(v_rev)
                determined.add(v_rev)

    for i in range(1, n + 1):
        if coords[i] is None:
            print("undecidable")
        else:
            print(coords[i][0], coords[i][1])

if __name__ == '__main__':
    solve()