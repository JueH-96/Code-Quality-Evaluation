import bisect
from collections import defaultdict

def merge_intervals(intervals):
    if not intervals:
        return []
    # Sort intervals based on the start of each interval
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = [list(sorted_intervals[0])]
    for current in sorted_intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1] + 1:
            # Merge the intervals
            last[1] = max(last[1], current[1])
        else:
            merged.append(list(current))
    # Convert back to tuples
    return [tuple(interval) for interval in merged]

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    ptr = 0
    N = int(data[ptr])
    ptr += 1
    M = int(data[ptr])
    ptr += 1
    Sx = int(data[ptr])
    ptr += 1
    Sy = int(data[ptr])
    ptr += 1

    houses = set()
    for _ in range(N):
        x = int(data[ptr])
        ptr += 1
        y = int(data[ptr])
        ptr += 1
        houses.add((x, y))

    horizontal_segments = defaultdict(list)
    vertical_segments = defaultdict(list)

    current_x, current_y = Sx, Sy

    for _ in range(M):
        D = data[ptr]
        ptr += 1
        C = int(data[ptr])
        ptr += 1

        if D == 'U':
            new_x = current_x
            new_y = current_y + C
        elif D == 'D':
            new_x = current_x
            new_y = current_y - C
        elif D == 'L':
            new_x = current_x - C
            new_y = current_y
        elif D == 'R':
            new_x = current_x + C
            new_y = current_y

        # Record the line segment
        if current_x == new_x:
            # Vertical movement
            x = current_x
            y_min = min(current_y, new_y)
            y_max = max(current_y, new_y)
            vertical_segments[x].append((y_min, y_max))
        else:
            # Horizontal movement
            y = current_y
            x_min = min(current_x, new_x)
            x_max = max(current_x, new_x)
            horizontal_segments[y].append((x_min, x_max))

        current_x, current_y = new_x, new_y

    # Merge intervals for horizontal and vertical segments
    horizontal_merged = dict()
    for y in horizontal_segments:
        intervals = horizontal_segments[y]
        merged = merge_intervals(intervals)
        if not merged:
            continue
        x_mins = [interval[0] for interval in merged]
        x_maxs = [interval[1] for interval in merged]
        horizontal_merged[y] = (x_mins, x_maxs)

    vertical_merged = dict()
    for x in vertical_segments:
        intervals = vertical_segments[x]
        merged = merge_intervals(intervals)
        if not merged:
            continue
        y_mins = [interval[0] for interval in merged]
        y_maxs = [interval[1] for interval in merged]
        vertical_merged[x] = (y_mins, y_maxs)

    # Check each house
    visited = set()
    for xh, yh in houses:
        # Check horizontal segments
        if yh in horizontal_merged:
            x_mins, x_maxs = horizontal_merged[yh]
            idx = bisect.bisect_right(x_mins, xh) - 1
            if idx >= 0 and xh <= x_maxs[idx]:
                visited.add((xh, yh))
                continue  # No need to check vertical if found in horizontal
        # Check vertical segments
        if xh in vertical_merged:
            y_mins, y_maxs = vertical_merged[xh]
            idx = bisect.bisect_right(y_mins, yh) - 1
            if idx >= 0 and yh <= y_maxs[idx]:
                visited.add((xh, yh))

    final_x, final_y = current_x, current_y
    count = len(visited)
    print(final_x, final_y, count)

if __name__ == '__main__':
    main()