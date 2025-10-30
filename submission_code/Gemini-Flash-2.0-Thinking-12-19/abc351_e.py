def solve():
    n = int(input())
    points = []
    for _ in range(n):
        x, y = map(int, input().split())
        points.append({'x': x, 'y': y})
    
    group1 = []
    group2 = []
    for p in points:
        if (p['x'] + p['y']) % 2 == 0:
            group1.append(p)
        else:
            group2.append(p)
            
    total_distance_sum = 0
    
    for group in [group1, group2]:
        m = len(group)
        if m < 2:
            continue
            
        z_values = sorted([p['x'] + p['y'] for p in group])
        s_z = 0
        for i in range(m):
            s_z += (2 * (i + 1) - 1 - m) * z_values[i]
            
        w_values = sorted([p['x'] - p['y'] for p in group])
        s_w = 0
        for i in range(m):
            s_w += (2 * (i + 1) - 1 - m) * w_values[i]
            
        group_distance_sum = (s_z + s_w) // 2
        total_distance_sum += group_distance_sum
        
    print(total_distance_sum)

if __name__ == '__main__':
    solve()