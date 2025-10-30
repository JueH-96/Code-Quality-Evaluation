import sys
import math
from itertools import permutations

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def main():
    N, S, T = map(int, sys.stdin.readline().split())
    segments = []
    for _ in range(N):
        A, B, C, D = map(int, sys.stdin.readline().split())
        segments.append(((A, B), (C, D)))
    
    # Generate all possible orders of segments
    min_time = float('inf')
    for order in permutations(range(N)):
        # For each order, try both starting points for each segment
        # We need to try all possible combinations of starting points
        # Since each segment has 2 choices, total 2^N combinations
        # For N up to 6, 2^6 = 64, which is manageable
        for mask in range(0, 2**N):
            current_time = 0
            current_pos = (0, 0)
            for i in range(N):
                seg_idx = order[i]
                seg = segments[seg_idx]
                # Determine which endpoint to start from based on the mask
                if (mask >> i) & 1:
                    start = seg[1]
                    end = seg[0]
                else:
                    start = seg[0]
                    end = seg[1]
                # Move to start without emitting laser
                dist = distance(current_pos, start)
                current_time += dist / S
                # Emit laser from start to end
                dist_seg = distance(start, end)
                current_time += dist_seg / T
                current_pos = end
            if current_time < min_time:
                min_time = current_time
    print("{0:.20f}".format(min_time))

if __name__ == "__main__":
    main()