import sys
import bisect
import collections

# Read all input data
data = sys.stdin.read().split()
index = 0

# Read N, M, S_x, S_y
N = int(data[index])
index += 1
M = int(data[index])
index += 1
S_x = int(data[index])
index += 1
S_y = int(data[index])
index += 1

# Read houses
houses = []
for _ in range(N):
    x = int(data[index])
    index += 1
    y = int(data[index])
    index += 1
    houses.append((x, y))

# Initialize current position
current_x = S_x
current_y = S_y

# Initialize intervals for horizontal and vertical moves
hor_intervals = collections.defaultdict(list)  # y -> list of [min_x, max_x]
ver_intervals = collections.defaultdict(list)  # x -> list of [min_y, max_y]

# Process each move and record intervals
for _ in range(M):
    D = data[index]  # Direction string
    index += 1
    C = int(data[index])  # Count
    index += 1
    if D == 'L' or D == 'R':  # Horizontal move
        if D == 'L':
            new_x = current_x - C
        else:  # D == 'R'
            new_x = current_x + C
        min_x_val = min(current_x, new_x)
        max_x_val = max(current_x, new_x)
        hor_intervals[current_y].append([min_x_val, max_x_val])
        current_x = new_x
    elif D == 'U' or D == 'D':  # Vertical move
        if D == 'U':
            new_y = current_y + C
        else:  # D == 'D'
            new_y = current_y - C
        min_y_val = min(current_y, new_y)
        max_y_val = max(current_y, new_y)
        ver_intervals[current_x].append([min_y_val, max_y_val])
        current_y = new_y

# Function to merge intervals
def merge_intervals(intervals_list):
    if not intervals_list:
        return []
    sorted_intervals = sorted(intervals_list, key=lambda x: x[0])  # Sort by start
    merged = []
    current_start = sorted_intervals[0][0]
    current_end = sorted_intervals[0][1]
    for start, end in sorted_intervals[1:]:
        if start <= current_end:  # Overlap or adjacent
            current_end = max(current_end, end)
        else:
            merged.append([current_start, current_end])
            current_start = start
            current_end = end
    merged.append([current_start, current_end])
    return merged

# Merge intervals for horizontal and vertical moves
hor_merged = {}
for y in hor_intervals:
    merged_list = merge_intervals(hor_intervals[y])
    if merged_list:
        hor_merged[y] = merged_list

ver_merged = {}
for x in ver_intervals:
    merged_list = merge_intervals(ver_intervals[x])
    if merged_list:
        ver_merged[x] = merged_list

# Function to check if point is in any interval
def point_in_intervals(point, intervals_list):
    if not intervals_list:
        return False
    starts = [iv[0] for iv in intervals_list]  # Extract start points
    idx = bisect.bisect_right(starts, point)
    if idx > 0 and intervals_list[idx - 1][0] <= point <= intervals_list[idx - 1][1]:
        return True
    return False

# Count the number of houses passed through or arrived at
count = 0
for house in houses:
    xh, yh = house
    if (yh in hor_merged and point_in_intervals(xh, hor_merged[yh])) or \
       (xh in ver_merged and point_in_intervals(yh, ver_merged[xh])):
        count += 1

# Output the final position and count
print(current_x, current_y, count)