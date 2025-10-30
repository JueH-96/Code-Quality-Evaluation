def solve():
    n, m, h, k = map(int, input().split())
    s = input()
    items = set()
    for _ in range(m):
        x, y = map(int, input().split())
        items.add((x, y))
    
    x, y = 0, 0
    for move in s:
        h -= 1
        if move == 'R':
            x += 1
        elif move == 'L':
            x -= 1
        elif move == 'U':
            y += 1
        else:
            y -= 1
        
        if h < 0:
            print("No")
            return
        
        if (x, y) in items and h < k:
            h = k
            items.remove((x, y))
    
    print("Yes")

solve()