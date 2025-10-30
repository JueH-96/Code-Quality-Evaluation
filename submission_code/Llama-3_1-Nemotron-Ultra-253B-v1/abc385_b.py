h, w, x, y = map(int, input().split())
x -= 1
y -= 1
grid = [input().strip() for _ in range(h)]
t = input().strip()

visited = set()
current_x, current_y = x, y

for direction in t:
    new_x, new_y = current_x, current_y
    if direction == 'U':
        new_x = current_x - 1
    elif direction == 'D':
        new_x = current_x + 1
    elif direction == 'L':
        new_y = current_y - 1
    elif direction == 'R':
        new_y = current_y + 1
    
    if 0 <= new_x < h and 0 <= new_y < w and grid[new_x][new_y] != '#':
        current_x, current_y = new_x, new_y
    
    if grid[current_x][current_y] == '@':
        visited.add((current_x, current_y))

final_x = current_x + 1
final_y = current_y + 1
count = len(visited)
print(final_x, final_y, count)