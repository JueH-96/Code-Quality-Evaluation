import math

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    n = int(data[0])
    points = [(0.0, 0.0)]
    index = 1
    for _ in range(n):
        x = int(data[index])
        y = int(data[index + 1])
        points.append((x, y))
        index += 2
    points.append((0.0, 0.0))
    
    total = 0.0
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        dx = x1 - x2
        dy = y1 - y2
        total += math.sqrt(dx * dx + dy * dy)
    
    print("{0:.20f}".format(total))

if __name__ == "__main__":
    main()