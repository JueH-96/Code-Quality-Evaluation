def solve():
    n = int(input())
    points = []
    for _ in range(n):
        x, y = map(int, input().split())
        points.append((x, y))
    
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            # Check if reachable: both sum and difference must have same parity
            if (x1 + y1) % 2 == (x2 + y2) % 2 and (x1 - y1) % 2 == (x2 - y2) % 2:
                # Distance is max of absolute differences
                dist = max(abs(x1 - x2), abs(y1 - y2))
                total += dist
    
    print(total)

solve()