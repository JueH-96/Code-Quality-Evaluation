import sys
from bisect import bisect_left, bisect_right
from collections import defaultdict

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr]); ptr += 1
    M = int(input[ptr]); ptr += 1
    Sx = int(input[ptr]); ptr += 1
    Sy = int(input[ptr]); ptr += 1

    houses = []
    for _ in range(N):
        x = int(input[ptr])
        y = int(input[ptr + 1])
        houses.append((x, y))
        ptr += 2

    movements = []
    for _ in range(M):
        D = input[ptr]
        C = int(input[ptr + 1])
        movements.append((D, C))
        ptr += 2

    current_x = Sx
    current_y = Sy

    vertical_ranges = defaultdict(list)
    horizontal_ranges = defaultdict(list)

    for D, C in movements:
        if D in ['U', 'D']:
            if D == 'U':
                new_y = current_y + C
            else:
                new_y = current_y - C
            x = current_x
            y_start = current_y
            y_end = new_y
            y_min = min(y_start, y_end)
            y_max = max(y_start, y_end)
            vertical_ranges[x].append((y_min, y_max))
            current_y = new_y
        else:
            if D == 'R':
                new_x = current_x + C
            else:
                new_x = current_x - C
            y = current_y
            x_start = current_x
            x_end = new_x
            x_min = min(x_start, x_end)
            x_max = max(x_start, x_end)
            horizontal_ranges[y].append((x_min, x_max))
            current_x = new_x

    def merge(ranges):
        if not ranges:
            return []
        ranges.sort()
        merged = [ranges[0]]
        for current in ranges[1:]:
            last = merged[-1]
            if current[0] <= last[1]:
                new_start = last[0]
                new_end = max(last[1], current[1])
                merged[-1] = (new_start, new_end)
            else:
                merged.append(current)
        return merged

    merged_vertical = {}
    for x in vertical_ranges:
        ranges = vertical_ranges[x]
        merged = merge(ranges)
        merged_vertical[x] = merged

    merged_horizontal = {}
    for y in horizontal_ranges:
        ranges = horizontal_ranges[y]
        merged = merge(ranges)
        merged_horizontal[y] = merged

    count = 0
    for X, Y in houses:
        found = False
        if X in merged_vertical:
            ranges = merged_vertical[X]
            low, high = 0, len(ranges) - 1
            while low <= high:
                mid = (low + high) // 2
                y_min, y_max = ranges[mid]
                if y_min > Y:
                    high = mid - 1
                else:
                    if Y <= y_max:
                        found = True
                        break
                    else:
                        low = mid + 1
            if found:
                count += 1
                continue

        if Y in merged_horizontal:
            ranges = merged_horizontal[Y]
            low, high = 0, len(ranges) - 1
            while low <= high:
                mid = (low + high) // 2
                x_min, x_max = ranges[mid]
                if x_min > X:
                    high = mid - 1
                else:
                    if X <= x_max:
                        found = True
                        break
                    else:
                        low = mid + 1
            if found:
                count += 1

    print(current_x, current_y, count)

if __name__ == "__main__":
    main()