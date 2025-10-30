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
        # For each segment, choose which endpoint to start from
        # There are 2^N possibilities
        for mask in range(0, 2**N):
            current_time = 0
            current_pos = (0, 0)
            for i in range(N):
                seg_idx = order[i]
                seg = segments[seg_idx]
                if mask & (1 << i):
                    start, end = seg[1], seg[0]
                else:
                    start, end = seg[0], seg[1]
                # Move to start
                dist = distance(current_pos, start)
                current_time += dist / S
                # Print the segment
                dist_seg = distance(start, end)
                current_time += dist_seg / T
                current_pos = end
            if current_time < min_time:
                min_time = current_time
    print("{0:.20f}".format(min_time))

if __name__ == "__main__":
    main()