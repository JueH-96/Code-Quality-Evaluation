# YOUR CODE HERE
def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    if M == 0:
        print("Yes")
        print("1 " * N)
        return
    
    from collections import defaultdict
    
    edges = []
    adj = defaultdict(list)
    
    index = 2
    for _ in range(M):
        u = int(data[index]) - 1
        v = int(data[index + 1]) - 1
        edges.append((u, v))
        adj[u].append(v)
        adj[v].append(u)
        index += 2
    
    # Check if the graph is bipartite
    color = [-1] * N
    
    def is_bipartite(v, c):
        color[v] = c
        for neighbor in adj[v]:
            if color[neighbor] == -1:
                if not is_bipartite(neighbor, 1 - c):
                    return False
            elif color[neighbor] == color[v]:
                return False
        return True
    
    bipartite = True
    for i in range(N):
        if color[i] == -1:
            if not is_bipartite(i, 0):
                bipartite = False
                break
    
    if not bipartite:
        print("No")
        return
    
    # If the graph is bipartite, we can assign values
    print("Yes")
    result = [0] * N
    for i in range(N):
        if color[i] == 0:
            result[i] = 1
        else:
            result[i] = 2
    
    print(" ".join(map(str, result)))

# To run the function, uncomment the following line:
# solve()