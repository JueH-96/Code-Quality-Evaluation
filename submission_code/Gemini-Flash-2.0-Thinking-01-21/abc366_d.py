def solve():
    n = int(input())
    a = []
    for _ in range(n):
        row_y = []
        for _ in range(n):
            row_z = list(map(int, input().split()))
            row_y.append(row_z)
        a.append(row_y)
    
    s = [[[0 for _ in range(n + 1)] for _ in range(n + 1)] for _ in range(n + 1)]
    
    for x in range(1, n + 1):
        for y in range(1, n + 1):
            for z in range(1, n + 1):
                s[x][y][z] = a[x-1][y-1][z-1] + s[x-1][y][z] + s[x][y-1][z] + s[x][z][z-1] - s[x-1][y-1][z] - s[x-1][z][z-1] - s[x][y-1][z-1] + s[x-1][y-1][z-1]
                
    q = int(input())
    results = []
    for _ in range(q):
        lx, rx, ly, ry, lz, rz = map(int, input().split())
        query_sum = (s[rx][ry][rz] - s[lx-1][ry][rz] - s[rx][ly-1][rz] - s[rx][ry][lz-1] 
                     + s[lx-1][ly-1][rz] + s[lx-1][ry][lz-1] + s[rx][ly-1][lz-1] - s[lx-1][ly-1][lz-1])
        results.append(query_sum)
        
    for result in results:
        print(result)

if __name__ == '__main__':
    solve()