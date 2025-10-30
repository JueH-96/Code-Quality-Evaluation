# YOUR CODE HERE
import sys
import math

def main():
    import sys
    import threading
    def solve():
        N = int(sys.stdin.readline())
        Q = list(map(int, sys.stdin.readline().split()))
        A = list(map(int, sys.stdin.readline().split()))
        B = list(map(int, sys.stdin.readline().split()))
        import itertools

        # Constraints are:
        # A_i * x + B_i * y <= Q_i
        # x >= 0, y >= 0

        # Collect constraints
        constraints = []
        N_constraints = []

        for i in range(N):
            A_i, B_i, Q_i = A[i], B[i], Q[i]
            constraints.append((A_i, B_i, Q_i))

        # Include x >= 0, y >= 0
        # constraints.append((-1, 0, 0))  # x >= 0
        # constraints.append((0, -1, 0))  # y >= 0

        # Collect all intersection points of constraint lines
        vertices = []

        # First, add the intersections with axes
        # x = 0
        x_0 = 0
        y_max_x0 = float('inf')
        for A_i, B_i, Q_i in constraints:
            if B_i == 0:
                if Q_i - A_i * x_0 < 0:
                    y_max_x0 = -1  # No solution
                    break
                else:
                    continue
            y_i = (Q_i - A_i * x_0) / B_i
            if y_i < 0:
                y_max_x0 = -1
                break
            y_max_x0 = min(y_max_x0, y_i)
        if y_max_x0 >= 0:
            vertices.append((x_0, y_max_x0))

        # y = 0
        y_0 = 0
        x_max_y0 = float('inf')
        for A_i, B_i, Q_i in constraints:
            if A_i == 0:
                if Q_i - B_i * y_0 < 0:
                    x_max_y0 = -1
                    break
                else:
                    continue
            x_i = (Q_i - B_i * y_0) / A_i
            if x_i < 0:
                x_max_y0 = -1
                break
            x_max_y0 = min(x_max_y0, x_i)
        if x_max_y0 >= 0:
            vertices.append((x_max_y0, y_0))

        # Intersection points between the constraints
        for i in range(len(constraints)):
            A1, B1, C1 = constraints[i]
            for j in range(i+1, len(constraints)):
                A2, B2, C2 = constraints[j]
                det = A1 * B2 - A2 * B1
                if det == 0:
                    continue  # Lines are parallel
                x = (C1 * B2 - C2 * B1) / det
                y = (A1 * C2 - A2 * C1) / det
                if x < -1 or y < -1:
                    continue
                # Check if point satisfies all constraints
                valid = True
                for A_i, B_i, Q_i in constraints:
                    if A_i * x + B_i * y - Q_i > 1e-6:
                        valid = False
                        break
                if valid and x >= -1e-6 and y >= -1e-6:
                    vertices.append((x, y))

        max_S = 0

        # For each vertex, check nearby integer points
        for x, y in vertices:
            x_start = max(0, int(math.floor(x)) - 5)
            x_end = int(math.ceil(x)) + 5
            y_start = max(0, int(math.floor(y)) - 5)
            y_end = int(math.ceil(y)) +5
            for xi in range(x_start, x_end + 1):
                for yi in range(y_start, y_end + 1):
                    if xi < 0 or yi < 0:
                        continue
                    valid = True
                    for k in range(N):
                        A_i, B_i, Q_i = A[k], B[k], Q[k]
                        total = A_i * xi + B_i * yi
                        if total > Q_i:
                            valid = False
                            break
                    if valid:
                        S = xi + yi
                        if S > max_S:
                            max_S = S

        print(int(max_S))
    threading.Thread(target=solve).start()

if __name__ == '__main__':
    main()