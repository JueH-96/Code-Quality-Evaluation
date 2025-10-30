def solve():
    n, m, sx, sy = map(int, input().split())
    houses = set()
    for _ in range(n):
        houses.add(tuple(map(int, input().split())))
    moves = []
    for _ in range(m):
        moves.append(input().split())

    current_x, current_y = sx, sy
    visited_houses = set()

    for direction, count_str in moves:
        count = int(count_str)
        nx, ny = current_x, current_y

        if direction == 'U':
            ny += count
            min_y = min(current_y, ny)
            max_y = max(current_y, ny)
            for hx, hy in houses:
                if hx == current_x and min_y < hy <= max_y:
                    visited_houses.add((hx, hy))
        elif direction == 'D':
            ny -= count
            min_y = min(current_y, ny)
            max_y = max(current_y, ny)
            for hx, hy in houses:
                if hx == current_x and max_y <= hy < min_y:
                    visited_houses.add((hx, hy))
        elif direction == 'L':
            nx -= count
            min_x = min(current_x, nx)
            max_x = max(current_x, nx)
            for hx, hy in houses:
                if hy == current_y and max_x <= hx < min_x:
                    visited_houses.add((hx, hy))
        elif direction == 'R':
            nx += count
            min_x = min(current_x, nx)
            max_x = max(current_x, nx)
            for hx, hy in houses:
                if hy == current_y and min_x < hx <= max_x:
                    visited_houses.add((hx, hy))
        current_x, current_y = nx, ny

    print(current_x, current_y, len(visited_houses))

if __name__ == '__main__':
    solve()