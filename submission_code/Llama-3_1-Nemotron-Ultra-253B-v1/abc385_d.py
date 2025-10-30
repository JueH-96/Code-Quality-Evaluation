import sys
from collections import defaultdict

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr]); ptr +=1
    M = int(input[ptr]); ptr +=1
    S_x = int(input[ptr]); ptr +=1
    S_y = int(input[ptr]); ptr +=1

    houses = []
    for _ in range(N):
        X = int(input[ptr]); ptr +=1
        Y = int(input[ptr]); ptr +=1
        houses.append((X, Y))

    h_segments = defaultdict(list)
    v_segments = defaultdict(list)

    current_x, current_y = S_x, S_y

    for _ in range(M):
        D = input[ptr]; ptr +=1
        C = int(input[ptr]); ptr +=1
        if D == 'U':
            new_y = current_y + C
            y_start, y_end = sorted((current_y, new_y))
            v_segments[current_x].append((y_start, y_end))
            current_y = new_y
        elif D == 'D':
            new_y = current_y - C
            y_start, y_end = sorted((new_y, current_y))
            v_segments[current_x].append((y_start, y_end))
            current_y = new_y
        elif D == 'L':
            new_x = current_x - C
            x_start, x_end = sorted((new_x, current_x))
            h_segments[current_y].append((x_start, x_end))
            current_x = new_x
        elif D == 'R':
            new_x = current_x + C
            x_start, x_end = sorted((current_x, new_x))
            h_segments[current_y].append((x_start, x_end))
            current_x = new_x

    # Merge intervals for horizontal segments
    for y in h_segments:
        intervals = h_segments[y]
        intervals.sort()
        merged = []
        for interval in intervals:
            if not merged:
                merged.append(interval)
            else:
                last = merged[-1]
                if interval[0] <= last[1]:
                    merged[-1] = (last[0], max(last[1], interval[1]))
                else:
                    merged.append(interval)
        h_segments[y] = merged

    # Merge intervals for vertical segments
    for x in v_segments:
        intervals = v_segments[x]
        intervals.sort()
        merged = []
        for interval in intervals:
            if not merged:
                merged.append(interval)
            else:
                last = merged[-1]
                if interval[0] <= last[1]:
                    merged[-1] = (last[0], max(last[1], interval[1]))
                else:
                    merged.append(interval)
        v_segments[x] = merged

    count = 0
    for X, Y in houses:
        found = False
        # Check horizontal
        if Y in h_segments:
            intervals = h_segments[Y]
            low, high = 0, len(intervals) - 1
            while low <= high and not found:
                mid = (low + high) // 2
                start, end = intervals[mid]
                if start > X:
                    high = mid - 1
                else:
                    if X <= end:
                        found = True
                    else:
                        low = mid + 1
        # Check vertical if not found
        if not found and X in v_segments:
            intervals = v_segments[X]
            low, high = 0, len(intervals) - 1
            while low <= high and not found:
                mid = (low + high) // 2
                start, end = intervals[mid]
                if start > Y:
                    high = mid - 1
                else:
                    if Y <= end:
                        found = True
                    else:
                        low = mid + 1
        if found:
            count += 1

    print(current_x, current_y, count)

if __name__ == '__main__':
    main()