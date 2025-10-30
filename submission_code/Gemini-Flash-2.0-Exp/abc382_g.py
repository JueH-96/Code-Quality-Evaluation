def solve():
    k, sx, sy, tx, ty = map(int, input().split())

    def get_tile(x, y):
        i = x // k
        j = y // k
        
        rem_x = x % k
        rem_y = y % k

        if (i % 2 == j % 2):
            return (i, j, rem_y)
        else:
            return (i, j, rem_x)

    s_tile = get_tile(sx, sy)
    t_tile = get_tile(tx, ty)

    s_i = sx // k
    s_j = sy // k
    t_i = tx // k
    t_j = ty // k

    print(abs(s_i - t_i) + abs(s_j - t_j))


t = int(input())
for _ in range(t):
    solve()