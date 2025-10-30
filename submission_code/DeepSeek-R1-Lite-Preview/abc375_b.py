import math

def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    points = []
    index = 1
    for _ in range(N):
        x = int(data[index])
        y = int(data[index + 1])
        points.append((x, y))
        index += 2
    total_cost = 0.0
    current = (0, 0)
    for point in points:
        dx = point[0] - current[0]
        dy = point[1] - current[1]
        distance = math.hypot(dx, dy)
        total_cost += distance
        current = point
    # Add the distance back to origin
    dx = 0 - current[0]
    dy = 0 - current[1]
    distance = math.hypot(dx, dy)
    total_cost += distance
    # Print with sufficient precision
    print("{0:.20f}".format(total_cost))

if __name__ == '__main__':
    main()