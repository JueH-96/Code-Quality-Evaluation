import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    n = int(input[ptr])
    ptr += 1
    m = int(input[ptr])
    ptr += 1
    sx = int(input[ptr])
    ptr += 1
    sy = int(input[ptr])
    ptr += 1

    houses = []
    for _ in range(n):
        x = int(input[ptr])
        ptr += 1
        y = int(input[ptr])
        ptr += 1
        houses.append((x, y))
    
    current_x, current_y = sx, sy
    horizontal_segments = []
    vertical_segments = []
    
    for _ in range(m):
        d = input[ptr]
        ptr += 1
        c = int(input[ptr])
        ptr += 1
        
        if d in ('L', 'R'):
            x_start = current_x
            if d == 'L':
                x_end = x_start - c
            else:
                x_end = x_start + c
            a = min(x_start, x_end)
            b = max(x_start, x_end)
            horizontal_segments.append((current_y, a, b))
            current_x = x_end
        else:
            y_start = current_y
            if d == 'D':
                y_end = y_start - c
            else:
                y_end = y_start + c
            a = min(y_start, y_end)
            b = max(y_start, y_end)
            vertical_segments.append((current_x, a, b))
            current_y = y_end
    
    # Merge horizontal intervals
    merged_horizontal = {}
    from collections import defaultdict
    h_dict = defaultdict(list)
    for y, a, b in horizontal_segments:
        h_dict[y].append((a, b))
    
    for y in h_dict:
        intervals = sorted(h_dict[y], key=lambda x: x[0])
        merged = []
        for interval in intervals:
            if not merged:
                merged.append(interval)
            else:
                last_a, last_b = merged[-1]
                curr_a, curr_b = interval
                if curr_a <= last_b:
                    merged[-1] = (last_a, max(last_b, curr_b))
                else:
                    merged.append(interval)
        a_list = [a for a, b in merged]
        merged_horizontal[y] = {'intervals': merged, 'a_list': a_list}
    
    # Merge vertical intervals
    merged_vertical = {}
    v_dict = defaultdict(list)
    for x, a, b in vertical_segments:
        v_dict[x].append((a, b))
    
    for x in v_dict:
        intervals = sorted(v_dict[x], key=lambda x: x[0])
        merged = []
        for interval in intervals:
            if not merged:
                merged.append(interval)
            else:
                last_a, last_b = merged[-1]
                curr_a, curr_b = interval
                if curr_a <= last_b:
                    merged[-1] = (last_a, max(last_b, curr_b))
                else:
                    merged.append(interval)
        a_list = [a for a, b in merged]
        merged_vertical[x] = {'intervals': merged, 'a_list': a_list}
    
    count = 0
    for (X, Y) in houses:
        found = False
        # Check horizontal
        if Y in merged_horizontal:
            data = merged_horizontal[Y]
            a_list = data['a_list']
            intervals = data['intervals']
            idx = bisect.bisect_right(a_list, X) - 1
            if idx >= 0 and idx < len(intervals):
                a, b = intervals[idx]
                if X <= b:
                    found = True
        # Check vertical
        if not found and X in merged_vertical:
            data = merged_vertical[X]
            a_list = data['a_list']
            intervals = data['intervals']
            idx = bisect.bisect_right(a_list, Y) - 1
            if idx >= 0 and idx < len(intervals):
                a, b = intervals[idx]
                if Y <= b:
                    found = True
        if found:
            count += 1
    
    print(current_x, current_y, count)

if __name__ == '__main__':
    main()