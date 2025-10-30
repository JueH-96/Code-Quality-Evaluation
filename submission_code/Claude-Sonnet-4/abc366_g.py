n, m = map(int, input().split())

if m == 0:
    print("Yes")
    print(" ".join(["1"] * n))
else:
    edges = []
    adj = [[] for _ in range(n)]
    
    for _ in range(m):
        u, v = map(int, input().split())
        u -= 1
        v -= 1
        edges.append((u, v))
        adj[u].append(v)
        adj[v].append(u)
    
    # Check if assigning same value to all vertices works
    # This works if every vertex has even degree
    all_even_degree = True
    for i in range(n):
        if len(adj[i]) % 2 == 1:
            all_even_degree = False
            break
    
    if all_even_degree:
        print("Yes")
        print(" ".join(["1"] * n))
    else:
        print("No")