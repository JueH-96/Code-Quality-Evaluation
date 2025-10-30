def solve():
    n = int(input())
    points = []
    for _ in range(n):
        x, y = map(int, input().split())
        points.append({'x': x, 'y': y})
    
    def get_distance(p1, p2):
        x1, y1 = p1['x'], p1['y']
        x2, y2 = p2['x'], p2['y']
        if (x1 + y1) % 2 != (x2 + y2) % 2:
            return 0
        else:
            dx = x2 - x1
            dy = y2 - y1
            return (abs(dx + dy) + abs(dx - dy)) // 2
            
    total_distance = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist = get_distance(points[i], points[j])
            total_distance += dist
            
    print(total_distance)

if __name__ == '__main__':
    solve()