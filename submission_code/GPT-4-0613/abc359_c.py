sx, sy, tx, ty = map(int, input().split())
dx, dy = abs(tx - sx), abs(ty - sy)
if dx < dy:
    dx, dy = dy, dx
    sx, sy, tx, ty = sy, sx, ty, tx
if sx <= tx:
    if sy <= ty:
        print(dx + dy + max(0, dy - dx + 1))
    else:
        print(dx + dy + max(0, dy - dx))
else:
    if sy >= ty:
        print(dx + dy + max(0, dy - dx + 1))
    else:
        print(dx + dy + max(0, dy - dx))