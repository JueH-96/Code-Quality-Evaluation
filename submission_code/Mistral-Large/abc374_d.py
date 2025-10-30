import itertools
import math
import sys

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def min_time_to_print_segments(N, S, T, segments):
    # Generate all permutations of the segments
    permutations = itertools.permutations(segments)

    min_time = float('inf')

    # Evaluate each permutation
    for perm in permutations:
        current_time = 0
        current_x, current_y = 0, 0

        for (a, b, c, d) in perm:
            # Move to the start of the segment without emitting the laser
            distance_to_start = calculate_distance(current_x, current_y, a, b)
            distance_to_end = calculate_distance(current_x, current_y, c, d)

            if distance_to_start < distance_to_end:
                current_time += distance_to_start / S
                current_x, current_y = a, b
                start_x, start_y = a, b
                end_x, end_y = c, d
            else:
                current_time += distance_to_end / S
                current_x, current_y = c, d
                start_x, start_y = c, d
                end_x, end_y = a, b

            # Emit the laser while moving along the segment
            current_time += calculate_distance(start_x, start_y, end_x, end_y) / T
            current_x, current_y = end_x, end_y

        min_time = min(min_time, current_time)

    return min_time

# Read input
input = sys.stdin.read
data = input().split()
N = int(data[0])
S = int(data[1])
T = int(data[2])

segments = []
index = 3
for _ in range(N):
    A = int(data[index])
    B = int(data[index + 1])
    C = int(data[index + 2])
    D = int(data[index + 3])
    segments.append((A, B, C, D))
    index += 4

# Calculate and print the minimum time
result = min_time_to_print_segments(N, S, T, segments)
print(result)