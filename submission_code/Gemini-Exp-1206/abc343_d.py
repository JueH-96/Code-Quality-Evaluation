def solve():
    n, t = map(int, input().split())
    updates = []
    for _ in range(t):
        updates.append(list(map(int, input().split())))
    
    scores = [0] * n
    
    for i in range(t):
        scores[updates[i][0]-1] += updates[i][1]
        print(len(set(scores)))

solve()