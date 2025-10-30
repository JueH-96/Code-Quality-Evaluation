def main():
    import sys
    import math

    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    total_cost = 0.0
    prev_x, prev_y = 0, 0
    index = 1
    for _ in range(N):
        x = int(data[index])
        y = int(data[index + 1])
        index += 2
        dx = x - prev_x
        dy = y - prev_y
        total_cost += math.sqrt(dx * dx + dy * dy)
        prev_x, prev_y = x, y

    # Add the cost to return back to the origin.
    total_cost += math.sqrt(prev_x * prev_x + prev_y * prev_y)
    
    # Print the answer with sufficient precision.
    sys.stdout.write("{:.12f}".format(total_cost))

if __name__ == '__main__':
    main()