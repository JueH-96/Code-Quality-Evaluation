import sys
import math

def main():
    input = sys.stdin.read().split()
    n = int(input[0])
    points = []
    idx = 1
    for _ in range(n):
        x = int(input[idx])
        y = int(input[idx+1])
        points.append((x, y))
        idx += 2
    
    total = 0.0
    prev_x, prev_y = 0, 0
    
    for x, y in points:
        dx = x - prev_x
        dy = y - prev_y
        dist = math.hypot(dx, dy)
        total += dist
        prev_x, prev_y = x, y
    
    # Return to origin
    dx = -prev_x
    dy = -prev_y
    dist = math.hypot(dx, dy)
    total += dist
    
    print("{0:.15f}".format(total))

if __name__ == "__main__":
    main()