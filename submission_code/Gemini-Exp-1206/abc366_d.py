def solve():
    n = int(input())
    a = []
    for _ in range(n):
        x_layer = []
        for _ in range(n):
            x_layer.append(list(map(int, input().split())))
        a.append(x_layer)
    
    q = int(input())
    for _ in range(q):
        lx, rx, ly, ry, lz, rz = map(int, input().split())
        total_sum = 0
        for x in range(lx - 1, rx):
            for y in range(ly - 1, ry):
                for z in range(lz - 1, rz):
                    total_sum += a[x][y][z]
        print(total_sum)

solve()