def solve():
    n, m, sx, sy = map(int, input().split())
    houses = set()
    for _ in range(n):
        x, y = map(int, input().split())
        houses.add((x, y))

    moves = []
    for _ in range(m):
        d, c = input().split()
        moves.append((d, int(c)))

    current_x, current_y = sx, sy
    visited_houses = set()

    for d, c in moves:
        start_x, start_y = current_x, current_y
        end_x, end_y = current_x, current_y

        if d == 'U':
            end_y += c
        elif d == 'D':
            end_y -= c
        elif d == 'L':
            end_x -= c
        elif d == 'R':
            end_x += c

        if start_x == end_x:
            for y in range(min(start_y, end_y), max(start_y, end_y) + 1):
                if (start_x, y) in houses:
                    visited_houses.add((start_x, y))
        elif start_y == end_y:
            for x in range(min(start_x, end_x), max(start_x, end_x) + 1):
                if (x, start_y) in houses:
                    visited_houses.add((x, start_y))

        current_x, current_y = end_x, end_y

    print(current_x, current_y, len(visited_houses))

if __name__ == "__main__":
    solve()