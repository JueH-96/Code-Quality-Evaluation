import math

def main():
    n = int(input().strip())
    points = []
    for _ in range(n):
        x, y = map(int, input().split())
        points.append((x, y))
    
    total = 0.0
    
    if n > 0:
        total += math.hypot(points[0][0], points[0][1])
        
        for i in range(1, n):
            dx = points[i][0] - points[i-1][0]
            dy = points[i][1] - points[i-1][1]
            total += math.hypot(dx, dy)
            
        total += math.hypot(points[-1][0], points[-1][1])
    
    print("{:.20f}".format(total))

if __name__ == "__main__":
    main()