def solve():
    k, sx, sy, tx, ty = map(int, input().split())

    start_x = sx + 0.5
    start_y = sy + 0.5
    target_x = tx + 0.5
    target_y = ty + 0.5

    i_start = int(start_x // k)
    j_start = int(start_y // k)
    i_target = int(target_x // k)
    j_target = int(target_y // k)

    print(abs(i_target - i_start) + abs(j_target - j_start))

t = int(input())
for _ in range(t):
    solve()