import bisect
from collections import defaultdict

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr]); ptr += 1
    M = int(input[ptr]); ptr += 1
    Sx = int(input[ptr]); ptr += 1
    Sy = int(input[ptr]); ptr += 1

    houses = []
    for _ in range(N):
        x = int(input[ptr]); ptr += 1
        y = int(input[ptr]); ptr += 1
        houses.append((x, y))

    vertical_segments = defaultdict(list)
    horizontal_segments = defaultdict(list)
    current_x, current_y = Sx, Sy

    for _ in range(M):
        D = input[ptr]; ptr += 1
        C = int(input[ptr]); ptr += 1

        if D == 'U':
            new_y = current_y + C
            y_start = min(current_y, new_y)
            y_end = max(current_y, new_y)
            key = current_x
            vertical_segments[key].append((y_start, y_end))
            current_y = new_y
        elif D == 'D':
            new_y = current_y - C
            y_start = min(current_y, new_y)
            y_end = max(current_y, new_y)
            key = current_x
            vertical_segments[key].append((y_start, y_end))
            current_y = new_y
        elif D == 'L':
            new_x = current_x - C
            x_start = min(current_x, new_x)
            x_end = max(current_x, new_x)
            key = current_y
            horizontal_segments[key].append((x_start, x_end))
            current_x = new_x
        elif D == 'R':
            new_x = current_x + C
            x_start = min(current_x, new_x)
            x_end = max(current_x, new_x)
            key = current_y
            horizontal_segments[key].append((x_start, x_end))
            current_x = new_x

    # Process vertical segments to merge intervals
    for x in vertical_segments:
        intervals = vertical_segments[x]
        intervals.sort()
        merged = []
        for interval in intervals:
            if not merged:
                merged.append(interval)
            else:
                last_a, last_b = merged[-1]
                a, b = interval
                if a <= last_b:
                    new_a = last_a
                    new_b = max(last_b, b)
                    merged.pop()
                    merged.append((new_a, new_b))
                else:
                    merged.append((a, b))
        vertical_segments[x] = merged

    # Process horizontal segments to merge intervals
    for y in horizontal_segments:
        intervals = horizontal_segments[y]
        intervals.sort()
        merged = []
        for interval in intervals:
            if not merged:
                merged.append(interval)
            else:
                last_a, last_b = merged[-1]
                a, b = interval
                if a <= last_b:
                    new_a = last_a
                    new_b = max(last_b, b)
                    merged.pop()
                    merged.append((new_a, new_b))
                else:
                    merged.append((a, b))
        horizontal_segments[y] = merged

    qualified = set()

    for (X, Y) in houses:
        # Check vertical segments
        if X in vertical_segments:
            intervals = vertical_segments[X]
            if intervals:
                starts = [a for (a, b) in intervals]
                idx = bisect.bisect_right(starts, Y) - 1
                if 0 <= idx < len(intervals):
                    a, b = intervals[idx]
                    if a <= Y <= b:
                        qualified.add((X, Y))
                        continue  # No need to check horizontal

        # Check horizontal segments
        if Y in horizontal_segments:
            intervals = horizontal_segments[Y]
            if intervals:
                starts = [a for (a, b) in intervals]
                idx = bisect.bisect_right(starts, X) - 1
                if 0 <= idx < len(intervals):
                    a, b = intervals[idx]
                    if a <= X <= b:
                        qualified.add((X, Y))

    count = len(qualified)
    print(f"{current_x} {current_y} {count}")

if __name__ == "__main__":
    main()