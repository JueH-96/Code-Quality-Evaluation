import math
from itertools import permutations

def main():
    N, S, T = map(int, input().split())
    segments = []
    for _ in range(N):
        A, B, C, D = map(int, input().split())
        segments.append(((A, B), (C, D)))
    
    min_time = float('inf')
    
    for perm in permutations(segments):
        for mask in range(1 << N):
            current = (0.0, 0.0)
            total = 0.0
            for i in range(N):
                seg = perm[i]
                bit = (mask >> i) & 1
                start = seg[bit]
                end = seg[1 - bit]
                # Move to start
                dx = start[0] - current[0]
                dy = start[1] - current[1]
                dist = math.hypot(dx, dy)
                total += dist / S
                # Print to end
                dx_p = end[0] - start[0]
                dy_p = end[1] - start[1]
                dist_p = math.hypot(dx_p, dy_p)
                total += dist_p / T
                current = end
            if total < min_time:
                min_time = total
    
    print("{0:.20f}".format(min_time))

if __name__ == "__main__":
    main()