import itertools
import math

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    S = int(input[idx])
    idx += 1
    T = int(input[idx])
    idx += 1

    lines = []
    for _ in range(N):
        A = int(input[idx])
        idx += 1
        B = int(input[idx])
        idx += 1
        C = int(input[idx])
        idx += 1
        D = int(input[idx])
        idx += 1
        point0 = (A, B)
        point1 = (C, D)
        dx = point0[0] - point1[0]
        dy = point0[1] - point1[1]
        length = math.hypot(dx, dy)
        lines.append((point0, point1, length))

    min_time = float('inf')

    for perm in itertools.permutations(lines):
        for dirs in itertools.product([0, 1], repeat=N):
            current_pos = (0.0, 0.0)
            total_time = 0.0
            for i in range(N):
                line = perm[i]
                point0, point1, line_length = line
                direction = dirs[i]
                if direction == 0:
                    start = point0
                    end = point1
                else:
                    start = point1
                    end = point0

                dx = start[0] - current_pos[0]
                dy = start[1] - current_pos[1]
                distance_move = math.hypot(dx, dy)
                total_time += distance_move / S

                total_time += line_length / T

                current_pos = end

            if total_time < min_time:
                min_time = total_time

    print("{0:.20f}".format(min_time))

if __name__ == '__main__':
    main()