def solve():
    n, m, x1 = map(int, input().split())
    trains = []
    for _ in range(m):
        trains.append(list(map(int, input().split())))
    
    x = [0] * m
    x[0] = x1
    
    changed = True
    while changed:
        changed = False
        for i in range(m):
            for j in range(m):
                if trains[i][1] == trains[j][0] and trains[i][3] <= trains[j][2]:
                    if trains[i][3] + x[i] > trains[j][2] + x[j]:
                        diff = (trains[i][3] + x[i]) - (trains[j][2] + x[j])
                        x[j] += diff
                        changed = True
    
    print(*x[1:])

solve()