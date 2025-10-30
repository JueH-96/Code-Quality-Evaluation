import math

def print_line_segments(N, S, T, A, B, C, D):
    total_time = 0
    current_x, current_y = 0, 0

    for i in range(N):
        # Move the laser position to the start of the line segment
        dx = A[i] - current_x
        dy = B[i] - current_y
        distance = math.sqrt(dx**2 + dy**2)
        move_time = distance / S
        total_time += move_time
        current_x, current_y = A[i], B[i]

        # Print the line segment
        dx = C[i] - A[i]
        dy = D[i] - B[i]
        distance = math.sqrt(dx**2 + dy**2)
        print_time = distance / T
        total_time += print_time
        current_x, current_y = C[i], D[i]

    return total_time

# Read input
N, S, T = map(int, input().split())
A, B, C, D = [], [], [], []
for _ in range(N):
    a, b, c, d = map(int, input().split())
    A.append(a)
    B.append(b)
    C.append(c)
    D.append(d)

# Solve the problem
total_time = print_line_segments(N, S, T, A, B, C, D)
print(f"{total_time:.15f}")