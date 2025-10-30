h, w = map(int, input().split())
si, sj = map(int, input().split())
grid = [input().strip() for _ in range(h)]
x = input().strip()

current_i, current_j = si, sj

for move in x:
    if move == 'L':
        new_i, new_j = current_i, current_j - 1
    elif move == 'R':
        new_i, new_j = current_i, current_j + 1
    elif move == 'U':
        new_i, new_j = current_i - 1, current_j
    elif move == 'D':
        new_i, new_j = current_i + 1, current_j
    
    if 1 <= new_i <= h and 1 <= new_j <= w:
        if grid[new_i - 1][new_j - 1] == '.':
            current_i, current_j = new_i, new_j

print(current_i, current_j)