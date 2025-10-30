import bisect
from collections import defaultdict

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    Sx = int(input[ptr])
    Sy = int(input[ptr + 1])
    ptr += 2
    
    houses = []
    for _ in range(N):
        x = int(input[ptr])
        y = int(input[ptr + 1])
        houses.append((x, y))
        ptr += 2
    
    movements = []
    for _ in range(M):
        d = input[ptr]
        c = int(input[ptr + 1])
        movements.append((d, c))
        ptr += 2
    
    current_x, current_y = Sx, Sy
    horizontal = defaultdict(list)  # key: y, value: list of (x_start, x_end)
    vertical = defaultdict(list)    # key: x, value: list of (y_start, y_end)
    
    for d, c in movements:
        if d == 'U':
            new_y = current_y + c
            y_start, y_end = current_y, new_y
            if y_start > y_end:
                y_start, y_end = y_end, y_start
            vertical[current_x].append((y_start, y_end))
            current_y = new_y
        elif d == 'D':
            new_y = current_y - c
            y_start, y_end = current_y, new_y
            if y_start > y_end:
                y_start, y_end = y_end, y_start
            vertical[current_x].append((y_start, y_end))
            current_y = new_y
        elif d == 'L':
            new_x = current_x - c
            x_start, x_end = current_x, new_x
            if x_start > x_end:
                x_start, x_end = x_end, x_start
            horizontal[current_y].append((x_start, x_end))
            current_x = new_x
        elif d == 'R':
            new_x = current_x + c
            x_start, x_end = current_x, new_x
            if x_start > x_end:
                x_start, x_end = x_end, x_start
            horizontal[current_y].append((x_start, x_end))
            current_x = new_x
    
    # Merge intervals for horizontal segments
    for y in list(horizontal.keys()):
        intervals = horizontal[y]
        intervals.sort(key=lambda x: x[0])
        merged = []
        for s, e in intervals:
            if not merged:
                merged.append((s, e))
            else:
                ls, le = merged[-1]
                if s <= le:
                    merged[-1] = (ls, max(le, e))
                else:
                    merged.append((s, e))
        starts = [s for s, e in merged]
        ends = [e for s, e in merged]
        horizontal[y] = (starts, ends)
    
    # Merge intervals for vertical segments
    for x in list(vertical.keys()):
        intervals = vertical[x]
        intervals.sort(key=lambda x: x[0])
        merged = []
        for s, e in intervals:
            if not merged:
                merged.append((s, e))
            else:
                ls, le = merged[-1]
                if s <= le:
                    merged[-1] = (ls, max(le, e))
                else:
                    merged.append((s, e))
        starts = [s for s, e in merged]
        ends = [e for s, e in merged]
        vertical[x] = (starts, ends)
    
    count = 0
    for x, y in houses:
        found = False
        if y in horizontal:
            starts, ends = horizontal[y]
            idx = bisect.bisect_right(starts, x) - 1
            if idx >= 0 and x <= ends[idx]:
                found = True
        if not found and x in vertical:
            starts, ends = vertical[x]
            idx = bisect.bisect_right(starts, y) - 1
            if idx >= 0 and y <= ends[idx]:
                found = True
        if found:
            count += 1
    
    print(current_x, current_y, count)

if __name__ == '__main__':
    main()