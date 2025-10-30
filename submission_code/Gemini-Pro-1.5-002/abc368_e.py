# YOUR CODE HERE
def solve():
    n, m, x1 = map(int, input().split())
    trains = []
    for i in range(m):
        a, b, s, t = map(int, input().split())
        trains.append((a, b, s, t, i))

    x = [0] * m
    x[0] = x1
    
    adj = [[] for i in range(n + 1)]
    for i in range(m):
        adj[trains[i][1]].append(i)

    for i in range(m):
        a, b, s, t, idx = trains[i]
        
        max_arrival = t + x[idx]
        
        for next_train_idx in adj[b]:
            next_a, next_b, next_s, next_t, next_idx = trains[next_train_idx]
            if t <= next_s:
                x[next_idx] = max(0, max_arrival - next_s + x[next_idx])
    
    ans = []
    for i in range(1,m):
        ans.append(x[i])
    print(*ans)

solve()