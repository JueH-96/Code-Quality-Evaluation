import sys
from collections import defaultdict

def main():
    input = sys.stdin.read().split()
    ptr = 0

    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    Sx = int(input[ptr])
    ptr += 1
    Sy = int(input[ptr])
    ptr += 1

    houses = []
    for _ in range(N):
        x = int(input[ptr])
        y = int(input[ptr + 1])
        houses.append((x, y))
        ptr += 2

    vertical_segments = defaultdict(list)
    horizontal_segments = defaultdict(list)
    x, y = Sx, Sy

    for _ in range(M):
        D = input[ptr]
        C = int(input[ptr + 1])
        ptr += 2
        if D == 'U':
            new_y = y + C
            y_min, y_max = min(y, new_y), max(y, new_y)
            vertical_segments[x].append((y_min, y_max))
            y = new_y
        elif D == 'D':
            new_y = y - C
            y_min, y_max = min(y, new_y), max(y, new_y)
            vertical_segments[x].append((y_min, y_max))
            y = new_y
        elif D == 'L':
            new_x = x - C
            x_min, x_max = min(x, new_x), max(x, new_x)
            horizontal_segments[y].append((x_min, x_max))
            x = new_x
        elif D == 'R':
            new_x = x + C
            x_min, x_max = min(x, new_x), max(x, new_x)
            horizontal_segments[y].append((x_min, x_max))
            x = new_x

    def merge(intervals):
        if not intervals:
            return []
        sorted_intervals = sorted(intervals)
        merged = [sorted_intervals[0]]
        for s, e in sorted_intervals[1:]:
            last_s, last_e = merged[-1]
            if s <= last_e:
                merged[-1] = (last_s, max(last_e, e))
            else:
                merged.append((s, e))
        return merged

    vertical_merged = {k: merge(v) for k, v in vertical_segments.items()}
    horizontal_merged = {k: merge(v) for k, v in horizontal_segments.items()}

    count = 0
    for X, Y in houses:
        found = False
        if X in vertical_merged:
            intervals = vertical_merged[X]
            left, right = 0, len(intervals) - 1
            while left <= right:
                mid = (left + right) // 2
                s, e = intervals[mid]
                if s <= Y <= e:
                    found = True
                    break
                elif Y < s:
                    right = mid - 1
                else:
                    left = mid + 1
            if found:
                count += 1
                continue
        if Y in horizontal_merged:
            intervals = horizontal_merged[Y]
            left, right = 0, len(intervals) - 1
            while left <= right:
                mid = (left + right) // 2
                s, e = intervals[mid]
                if s <= X <= e:
                    found = True
                    break
                elif X < s:
                    right = mid - 1
                else:
                    left = mid + 1
            if found:
                count += 1

    print(f"{x} {y} {count}")

if __name__ == "__main__":
    main()