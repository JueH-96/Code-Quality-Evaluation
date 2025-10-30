def solve():
    n, d = map(int, input().split())
    coords = []
    for _ in range(n):
        coords.append(list(map(int, input().split())))
    
    infected = [False] * n
    infected[0] = True
    
    changed = True
    while changed:
        changed = False
        for i in range(n):
            if infected[i]:
                for j in range(n):
                    if not infected[j]:
                        dist = ((coords[i][0] - coords[j][0])**2 + (coords[i][1] - coords[j][1])**2)**0.5
                        if dist <= d:
                            infected[j] = True
                            changed = True
    
    for i in range(n):
        if infected[i]:
            print("Yes")
        else:
            print("No")

solve()