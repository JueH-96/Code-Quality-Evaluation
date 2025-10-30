def main():
    import sys
    import math

    data = sys.stdin.read().strip().split()
    N = int(data[0])
    coords = [(int(data[i*2+1]), int(data[i*2+2])) for i in range(N)]

    total_distance = 0.0

    # Start from the origin
    prev_x, prev_y = 0, 0

    # Visit each point in order
    for x, y in coords:
        dx = x - prev_x
        dy = y - prev_y
        total_distance += math.sqrt(dx*dx + dy*dy)
        prev_x, prev_y = x, y

    # Return to the origin
    dx = -prev_x
    dy = -prev_y
    total_distance += math.sqrt(dx*dx + dy*dy)

    print(total_distance)

# IMPORTANT: Do not forget to call main()
main()