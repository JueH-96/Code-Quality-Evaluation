H, W, N = map(int, input().split())

socket = ['.'*W] * H

TAKAHASHI = (0, 0)

dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]
pos = dirs[0]

MOVES = {
    (0, 0): lambda cur_dir: (cur_dir[1], -cur_dir[0]),  # White局面の次の向き
    (0, 1): lambda cur_dir: (-cur_dir[1], cur_dir[0]),  # Black局面の次の向き
}

# TAKAHASHIが進むべき次のマスの位置
def next_pos(y, x, cur_dir):
    ny = ((y + cur_dir[0]) % H)
    nx = ((x + cur_dir[1]) % W)
    return (ny, nx)

for n in range(N):
    y, x = TAKAHASHI
    if socket[y][x] == '.':
        socket[y] = socket[y][:x] + '#' + socket[y][x+1:]
        cur_dir = MOVES[(0, 0)](pos)
    else:
        socket[y] = socket[y][:x] + '.' + socket[y][x+1:]
        cur_dir = MOVES[(0, 1)](pos)

    TAKAHASHI = next_pos(y, x, cur_dir)
    pos = cur_dir

for row in socket:
    print(row)