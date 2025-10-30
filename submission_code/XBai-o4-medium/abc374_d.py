import itertools
import math
import sys

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx]); idx += 1
    S = int(input[idx]); idx += 1
    T = int(input[idx]); idx += 1
    segs = []
    for _ in range(N):
        A = int(input[idx]); idx += 1
        B = int(input[idx]); idx += 1
        C = int(input[idx]); idx += 1
        D = int(input[idx]); idx += 1
        segs.append(((A, B), (C, D)))
    
    min_time = float('inf')
    
    from itertools import permutations, product
    
    for perm in permutations(segs):
        for directions in product([False, True], repeat=N):
            current_x = 0.0
            current_y = 0.0
            total_time = 0.0
            for i in range(N):
                seg = perm[i]
                if directions[i]:
                    start = seg[1]
                    end = seg[0]
                else:
                    start = seg[0]
                    end = seg[1]
                # Calculate movement time
                dx = start[0] - current_x
                dy = start[1] - current_y
                move_dist = math.hypot(dx, dy)
                move_time = move_dist / S
                total_time += move_time
                # Calculate emission time
                emit_dist = math.hypot(end[0] - start[0], end[1] - start[1])
                emit_time = emit_dist / T
                total_time += emit_time
                # Update current position
                current_x = end[0]
                current_y = end[1]
            if total_time < min_time:
                min_time = total_time
    print("{0:.20f}".format(min_time))

if __name__ == '__main__':
    main()