import itertools
import math

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    S = int(data[idx])
    idx += 1
    T = int(data[idx])
    idx += 1

    segments = []
    lengths = []
    for _ in range(N):
        a = int(data[idx])
        idx += 1
        b = int(data[idx])
        idx += 1
        c = int(data[idx])
        idx += 1
        d = int(data[idx])
        idx += 1
        segments.append(((a, b), (c, d)))
        dx = c - a
        dy = d - b
        len_seg = math.hypot(dx, dy)
        lengths.append(len_seg)
    total_print_time = sum(lengths) / T

    min_time = float('inf')

    for perm in itertools.permutations(range(N)):
        for mask in range(1 << N):
            current_x, current_y = 0.0, 0.0
            movement_time = 0.0
            for i in range(N):
                seg_index = perm[i]
                direction = (mask >> i) & 1
                if direction == 0:
                    start_x, start_y = segments[seg_index][0]
                    end_x, end_y = segments[seg_index][1]
                else:
                    start_x, start_y = segments[seg_index][1]
                    end_x, end_y = segments[seg_index][0]
                dx = start_x - current_x
                dy = start_y - current_y
                dist = math.hypot(dx, dy)
                movement_time += dist / S
                current_x, current_y = end_x, end_y
            total_time = total_print_time + movement_time
            if total_time < min_time:
                min_time = total_time

    print("{:.20f}".format(min_time))

if __name__ == '__main__':
    main()