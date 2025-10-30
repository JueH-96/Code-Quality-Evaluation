def solve():
    n, m, sx, sy = map(int, input().split())
    houses = []
    for _ in range(n):
        houses.append(tuple(map(int, input().split())))
    
    moves = []
    for _ in range(m):
        d, c = input().split()
        moves.append((d, int(c)))
    
    visited_houses = set()
    
    for direction, distance in moves:
        px, py = sx, sy
        
        if direction == 'U':
            sy += distance
        elif direction == 'D':
            sy -= distance
        elif direction == 'L':
            sx -= distance
        elif direction == 'R':
            sx += distance
        
        for hx, hy in houses:
            
            if direction == 'U':
                if px == hx and py < hy <= sy:
                    visited_houses.add((hx, hy))
            elif direction == 'D':
                if px == hx and sy <= hy < py:
                    visited_houses.add((hx, hy))
            elif direction == 'L':
                if py == hy and sx <= hx < px:
                    visited_houses.add((hx, hy))
            elif direction == 'R':
                if py == hy and px < hx <= sx:
                    visited_houses.add((hx, hy))
        
    print(sx, sy, len(visited_houses))

solve()