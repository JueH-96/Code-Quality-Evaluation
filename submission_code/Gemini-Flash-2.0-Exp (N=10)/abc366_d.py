def solve():
    n = int(input())
    a = []
    for _ in range(n):
        for _ in range(n):
            row = list(map(int, input().split()))
            a.append(row)
    
    q = int(input())
    for _ in range(q):
        lx, rx, ly, ry, lz, rz = map(int, input().split())
        lx -= 1
        ly -= 1
        lz -= 1
        
        total_sum = 0
        for x in range(lx, rx):
            for y in range(ly, ry):
                for z in range(lz, rz):
                    total_sum += a[x * n + y][z]
        print(total_sum)

solve()