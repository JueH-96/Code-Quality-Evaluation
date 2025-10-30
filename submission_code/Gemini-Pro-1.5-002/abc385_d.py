def solve():
    n, m, sx, sy = map(int, input().split())
    houses = []
    for _ in range(n):
        houses.append(tuple(map(int, input().split())))
    
    moves = []
    for _ in range(m):
        moves.append(input().split())
    
    visited_houses = set()
    current_x = sx
    current_y = sy
    
    for move in moves:
        direction = move[0]
        distance = int(move[1])
        
        next_x = current_x
        next_y = current_y
        
        if direction == 'U':
            next_y += distance
        elif direction == 'D':
            next_y -= distance
        elif direction == 'L':
            next_x -= distance
        elif direction == 'R':
            next_x += distance
            
        for house in houses:
            hx, hy = house
            
            if direction == 'U' or direction == 'D':
                if hx == current_x:
                    if min(current_y, next_y) <= hy <= max(current_y, next_y):
                        visited_houses.add(house)
            elif direction == 'L' or direction == 'R':
                if hy == current_y:
                    if min(current_x, next_x) <= hx <= max(current_x, next_x):
                        visited_houses.add(house)
        
        current_x = next_x
        current_y = next_y
        
    print(current_x, current_y, len(visited_houses))

solve()