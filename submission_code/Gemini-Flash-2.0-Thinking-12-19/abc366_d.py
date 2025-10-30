def solve():
    n = int(input())
    a_3d = []
    for _ in range(n):
        y_slices = []
        for _ in range(n):
            row_values = list(map(int, input().split()))
            y_slices.append(row_values)
        a_3d.append(y_slices)
    
    prefix_sum_3d = [[[0] * n for _ in range(n)] for _ in range(n)]
    
    for x in range(n):
        for y in range(n):
            for z in range(n):
                prefix_sum_3d[x][y][z] = a_3d[x][y][z]
                if x > 0:
                    prefix_sum_3d[x][y][z] += prefix_sum_3d[x-1][y][z]
                if y > 0:
                    prefix_sum_3d[x][y][z] += prefix_sum_3d[x][y-1][z]
                if z > 0:
                    prefix_sum_3d[x][y][z] += prefix_sum_3d[x][y][z-1]
                if x > 0 and y > 0:
                    prefix_sum_3d[x][y][z] -= prefix_sum_3d[x-1][y-1][z]
                if x > 0 and z > 0:
                    prefix_sum_3d[x][y][z] -= prefix_sum_3d[x-1][y][z-1]
                if y > 0 and z > 0:
                    prefix_sum_3d[x][y][z] -= prefix_sum_3d[x][y-1][z-1]
                if x > 0 and y > 0 and z > 0:
                    prefix_sum_3d[x][y][z] += prefix_sum_3d[x-1][y-1][z-1]
                    
    q = int(input())
    results = []
    for _ in range(q):
        lx, rx, ly, ry, lz, rz = map(int, input().split())
        lx -= 1
        rx -= 1
        ly -= 1
        ry -= 1
        lz -= 1
        rz -= 1
        
        def get_prefix_sum(x_idx, y_idx, z_idx):
            if x_idx < 0 or y_idx < 0 or z_idx < 0:
                return 0
            return prefix_sum_3d[x_idx][y_idx][z_idx]
            
        total_sum = get_prefix_sum(rx, ry, rz) - get_prefix_sum(lx - 1, ry, rz) - get_prefix_sum(rx, ly - 1, rz) - get_prefix_sum(rx, ry, lz - 1) + get_prefix_sum(lx - 1, ly - 1, rz) + get_prefix_sum(lx - 1, ry, lz - 1) + get_prefix_sum(rx, ly - 1, lz - 1) - get_prefix_sum(lx - 1, ly - 1, lz - 1)
        results.append(total_sum)
        
    for result in results:
        print(result)

if __name__ == '__main__':
    solve()