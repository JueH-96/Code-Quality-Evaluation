import math
import itertools
import sys

# Read input from stdin
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
S = int(data[index])  # Speed without emitting laser
index += 1
T = int(data[index])  # Speed while emitting laser
index += 1

# Read segments and store endpoints as floats
segments = []
for _ in range(N):
    A = float(data[index])
    index += 1
    B = float(data[index])
    index += 1
    C = float(data[index])
    index += 1
    D = float(data[index])
    index += 1
    segments.append(((A, B), (C, D)))

# Compute the fixed total printing distance
D_print = 0.0
for seg in segments:
    p1, p2 = seg
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    len_seg = math.sqrt(dx**2 + dy**2)
    D_print += len_seg

# Generate all permutations of segment orders and all direction combinations
perms = list(itertools.permutations(range(N)))
dirs = list(itertools.product([0, 1], repeat=N))

# Initialize minimum time to infinity
min_time = float('inf')

# Iterate over all possible orders and directions
for perm in perms:
    for dir_tuple in dirs:
        # Start from (0, 0)
        cur_pos = (0.0, 0.0)
        D_move = 0.0  # Total distance moved without emitting laser
        # For each segment in the current order
        for k in range(N):
            seg_idx = perm[k]  # Index of the segment
            dir_seg = dir_tuple[seg_idx]  # Direction for the segment (0 or 1)
            seg = segments[seg_idx]
            if dir_seg == 0:
                start_pt = seg[0]
                end_pt = seg[1]
            else:
                start_pt = seg[1]
                end_pt = seg[0]
            # Calculate distance to move to start point without emitting
            dx_move = cur_pos[0] - start_pt[0]
            dy_move = cur_pos[1] - start_pt[1]
            dist_move_val = math.sqrt(dx_move**2 + dy_move**2)
            D_move += dist_move_val
            # After printing, update position to end point
            cur_pos = end_pt
        # Calculate total time for this sequence
        total_time = (D_move / S) + (D_print / T)
        # Update minimum time if smaller
        if total_time < min_time:
            min_time = total_time

# Output the minimum time with 10 decimal places
print("{:.10f}".format(min_time))