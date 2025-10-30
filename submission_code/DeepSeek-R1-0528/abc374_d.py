import math
import itertools
import sys

def main():
    data = sys.stdin.read().strip().split()
    if not data:
        print(0.0)
        return
    n = int(data[0])
    S_val = int(data[1])
    T_val = int(data[2])
    segs = []
    index = 3
    base_time = 0.0
    for i in range(n):
        a = int(data[index])
        b = int(data[index+1])
        c = int(data[index+2])
        d = int(data[index+3])
        index += 4
        segs.append((a, b, c, d))
        dx = c - a
        dy = d - b
        base_time += math.sqrt(dx*dx + dy*dy) / T_val

    if n == 0:
        print(0.0)
        return

    min_total = float('inf')

    for perm in itertools.permutations(segs):
        for mask in range(1 << n):
            x, y = 0.0, 0.0
            non_emit_time = 0.0
            for i in range(n):
                a, b, c, d = perm[i]
                if (mask >> i) & 1:
                    start_pt = (c, d)
                    end_pt = (a, b)
                else:
                    start_pt = (a, b)
                    end_pt = (c, d)
                dx_move = start_pt[0] - x
                dy_move = start_pt[1] - y
                dist = math.sqrt(dx_move*dx_move + dy_move*dy_move)
                non_emit_time += dist / S_val
                x, y = end_pt
            total_time = base_time + non_emit_time
            if total_time < min_total:
                min_total = total_time
    print(min_total)

if __name__ == "__main__":
    main()