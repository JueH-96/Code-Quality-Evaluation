N = int(input())
buildings = []
for i in range(N):
    x, h = map(int, input().split())
    buildings.append((x, h))

if N == 1:
    print(-1)
else:
    max_height = 0
    for i in range(N):
        x1, h1 = buildings[i]
        max_h = 0
        for j in range(i + 1, N):
            x2, h2 = buildings[j]
            h = (h1 * x2 + h2 * x1) / (x2 - x1)
            max_h = max(max_h, h)
        max_height = max(max_height, max_h)
    
    possible = True
    for i in range(N):
        x,h = buildings[i]
        if h/x < max_height:
            possible = False
            break
    if possible:
        print(-1)
    else:
        print(max_height)