# YOUR CODE HERE
import sys
import math

def main():
    import threading
    def solve():
        import sys
        input = sys.stdin.readline
        N = int(sys.stdin.readline())
        points = []
        for _ in range(N):
            x_str, y_str = sys.stdin.readline().split()
            x = int(x_str)
            y = int(y_str)
            points.append((x, y))

        total_cost = 0.0
        prev_x, prev_y = 0, 0  # Start at origin
        for x, y in points:
            delta_x = x - prev_x
            delta_y = y - prev_y
            dist = math.hypot(delta_x, delta_y)
            total_cost += dist
            prev_x, prev_y = x, y
        # Return to origin
        delta_x = 0 - prev_x
        delta_y = 0 - prev_y
        dist = math.hypot(delta_x, delta_y)
        total_cost += dist

        print('{0:.20f}'.format(total_cost))

    threading.Thread(target=solve,).start()

if __name__ == '__main__':
    main()