import math

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    n = int(input[idx])
    idx += 1
    points = []
    for _ in range(n):
        x = int(input[idx])
        y = int(input[idx+1])
        points.append((x, y))
        idx += 2
    
    path = [(0, 0)]
    path.extend(points)
    path.append((0, 0))
    
    total = 0.0
    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i+1]
        dx = x2 - x1
        dy = y2 - y1
        total += math.hypot(dx, dy)
    
    print("{0:.20f}".format(total))

if __name__ == "__main__":
    main()