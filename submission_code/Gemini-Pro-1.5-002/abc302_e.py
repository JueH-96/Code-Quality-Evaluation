# YOUR CODE HERE
def solve():
    n, q = map(int, input().split())
    adj = [set() for _ in range(n)]
    isolated = n

    for _ in range(q):
        query = list(map(int, input().split()))
        
        if query[0] == 1:
            u, v = query[1] - 1, query[2] - 1
            adj[u].add(v)
            adj[v].add(u)
            if len(adj[u]) == 1:
                isolated -= 1
            if len(adj[v]) == 1:
                isolated -= 1
        else:
            v = query[1] - 1
            
            for neighbor in list(adj[v]):
                if len(adj[neighbor]) == 1:
                    isolated +=1
                adj[neighbor].remove(v)
            
            if len(adj[v]) > 0:
                isolated += len(adj[v]) - (0 if len(adj[v]) == 0 else 0)

            adj[v].clear()
            if len(adj[v]) == 0:
                isolated = isolated - (0 if len(adj[v]) > 0 else -1)
                
        print(isolated)

solve()