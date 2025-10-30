def solve():
    n, m, sx, sy = map(int, input().split())
    houses = []
    for _ in range(n):
        houses.append(tuple(map(int, input().split())))
    
    moves = []
    for _ in range(m):
        d, c = input().split()
        moves.append((d, int(c)))
    
    curr_x, curr_y = sx, sy
    visited_houses = set()
    
    for d, c in moves:
        next_x, next_y = curr_x, curr_y
        if d == 'U':
            next_y += c
        elif d == 'D':
            next_y -= c
        elif d == 'L':
            next_x -= c
        elif d == 'R':
            next_x += c
        
        for hx, hy in houses:
            if (curr_x == next_x == hx and min(curr_y, next_y) <= hy <= max(curr_y, next_y)) or \
               (curr_y == next_y == hy and min(curr_x, next_x) <= hx <= max(curr_x, next_x)):
                visited_houses.add((hx, hy))
        
        curr_x, curr_y = next_x, next_y
    
    print(curr_x, curr_y, len(visited_houses))

solve()