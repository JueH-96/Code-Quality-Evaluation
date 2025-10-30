def solve():
    n, m, x1 = map(int, input().split())
    trains = []
    for _ in range(m):
        trains.append(list(map(int, input().split())))
    
    x = [0] * m
    x[0] = x1
    
    adj = [[] for _ in range(m)]
    for i in range(m):
        for j in range(m):
            if i != j and trains[i][1] == trains[j][0] and trains[i][3] <= trains[j][2]:
                adj[i].append(j)
    
    q = [0]
    while q:
        curr = q.pop(0)
        for neighbor in adj[curr]:
            new_x = max(0, trains[curr][3] + x[curr] - trains[neighbor][2])
            if new_x > x[neighbor]:
                x[neighbor] = new_x
                q.append(neighbor)

    print(*x[1:])

solve()