h, w = map(int, input().split())

si, sj = map(lambda x: int(x) - 1, input().split())

maze = []
for i in range(h):
    maze.append(list(input()))

cmd = input()

dx = [0, 0, 1, -1]
dy = [1, -1, 0, 0]

def in_maze(x, y):
    return 0 <= x < h and 0 <= y < w

cmd_map = {'L': 0, 'R': 1, 'D': 2, 'U': 3}
for char in cmd:
    ix = cmd_map[char]
    nx = si + dx[ix]
    ny = sj + dy[ix]
    if in_maze(nx, ny) and maze[nx][ny] == '.':
        si, sj = nx, ny

print(si + 1, sj + 1)