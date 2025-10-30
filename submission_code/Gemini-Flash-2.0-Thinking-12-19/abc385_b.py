def solve():
    h, w, x, y = map(int, input().split())
    grid = [input() for _ in range(h)]
    t = input()

    current_r, current_c = x - 1, y - 1
    visited_houses = set()

    if grid[current_r][current_c] == '@':
        visited_houses.add((current_r, current_c))

    for move in t:
        next_r, next_c = current_r, current_c
        if move == 'U':
            next_r -= 1
        elif move == 'D':
            next_r += 1
        elif move == 'L':
            next_c -= 1
        elif move == 'R':
            next_c += 1

        if 0 <= next_r < h and 0 <= next_c < w and grid[next_r][next_c] != '#':
            current_r, current_c = next_r, next_c
            if grid[current_r][current_c] == '@':
                visited_houses.add((current_r, current_c))

    print(current_r + 1, current_c + 1, len(visited_houses))

if __name__ == '__main__':
    solve()