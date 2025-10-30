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
        if direction == 'U':
            for i in range(1, count + 1):
                next_y = current_y + i
                if (current_x, next_y) in houses:
                    visited_houses.add((current_x, next_y))
            current_y += count
        elif direction == 'D':
            for i in range(1, count + 1):
                next_y = current_y - i
                if (current_x, next_y) in houses:
                    visited_houses.add((current_x, next_y))
            current_y -= count
        elif direction == 'L':
            for i in range(1, count + 1):
                next_x = current_x - i
                if (next_x, current_y) in houses:
                    visited_houses.add((next_x, current_y))
            current_x -= count
        elif direction == 'R':
            for i in range(1, count + 1):
                next_x = current_x + i
                if (next_x, current_y) in houses:
                    visited_houses.add((next_x, current_y))
            current_x += count

    print(current_x, current_y, len(visited_houses))

if __name__ == "__main__":
    solve()