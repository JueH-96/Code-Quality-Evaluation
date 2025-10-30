import sys

def main():
    n = int(sys.stdin.readline())
    points = [tuple(map(int, sys.stdin.readline().split())) for _ in range(n)]
    
    prev_x, prev_y = 0, 0
    total = 0.0
    
    for x, y in points:
        dx = x - prev_x
        dy = y - prev_y
        total += (dx ** 2 + dy ** 2) ** 0.5
        prev_x, prev_y = x, y
    
    # Add the distance from the last point back to the origin
    dx = -prev_x
    dy = -prev_y
    total += (dx ** 2 + dy ** 2) ** 0.5
    
    # Print with sufficient precision
    print("{0:.20f}".format(total))

if __name__ == "__main__":
    main()