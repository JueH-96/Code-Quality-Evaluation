def solve():
    n, m, sx, sy = map(int, input().split())
    houses = []
    for _ in range(n):
        houses.append(tuple(map(int, input().split())))
    
    moves = []
    for _ in range(m):
        moves.append(input().split())

    visited_houses = set()
    curr_x, curr_y = sx, sy

    for move_dir, move_dist in moves:
        move_dist = int(move_dist)
        
        if move_dir == 'U':
            for i in range(1, move_dist + 1):
                next_x, next_y = curr_x, curr_y + i
                if (next_x, next_y) in houses:
                    visited_houses.add((next_x, next_y))
            curr_x, curr_y = curr_x, curr_y + move_dist
        elif move_dir == 'D':
            for i in range(1, move_dist + 1):
                next_x, next_y = curr_x, curr_y - i
                if (next_x, next_y) in houses:
                    visited_houses.add((next_x, next_y))
            curr_x, curr_y = curr_x, curr_y - move_dist
        elif move_dir == 'L':
            for i in range(1, move_dist + 1):
                next_x, next_y = curr_x - i, curr_y
                if (next_x, next_y) in houses:
                    visited_houses.add((next_x, next_y))
            curr_x, curr_y = curr_x - move_dist, curr_y
        elif move_dir == 'R':
            for i in range(1, move_dist + 1):
                next_x, next_y = curr_x + i, curr_y
                if (next_x, next_y) in houses:
                    visited_houses.add((next_x, next_y))
            curr_x, curr_y = curr_x + move_dist, curr_y
            
    print(curr_x, curr_y, len(visited_houses))

solve()