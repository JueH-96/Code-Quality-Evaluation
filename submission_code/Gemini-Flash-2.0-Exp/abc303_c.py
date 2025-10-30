def solve():
    n, m, h, k = map(int, input().split())
    s = input()
    items = []
    for _ in range(m):
        items.append(tuple(map(int, input().split())))

    x, y = 0, 0
    for move in s:
        if move == 'R':
            x += 1
        elif move == 'L':
            x -= 1
        elif move == 'U':
            y += 1
        else:
            y -= 1
        
        h -= 1
        
        if h < 0:
            print("No")
            return
        
        if (x, y) in items and h < k:
            h = k
            items.remove((x, y))
            
    print("Yes")

solve()