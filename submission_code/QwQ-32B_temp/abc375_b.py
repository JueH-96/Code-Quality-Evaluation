import math

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    points = []
    idx = 1
    for i in range(N):
        x = int(input[idx])
        y = int(input[idx+1])
        points.append((x, y))
        idx += 2
    
    total = 0.0
    prev_x, prev_y = 0, 0
    
    for x, y in points:
        dx = x - prev_x
        dy = y - prev_y
        distance = (dx**2 + dy**2) ** 0.5
        total += distance
        prev_x, prev_y = x, y
    
    # Return to origin
    dx = 0 - prev_x
    dy = 0 - prev_y
    distance = (dx**2 + dy**2) ** 0.5
    total += distance
    
    print("{0:.15f}".format(total))

if __name__ == "__main__":
    main()