def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    rectangles = []
    idx = 1
    for _ in range(N):
        A = int(data[idx])
        B = int(data[idx+1])
        C = int(data[idx+2])
        D = int(data[idx+3])
        rectangles.append((A, B, C, D))
        idx += 4
    
    # Create events
    events = []
    for rect in rectangles:
        A, B, C, D = rect
        events.append((A, 'start', C, D))
        events.append((B, 'end', C, D))
    
    # Sort events: by x, and for same x, 'start' before 'end'
    events.sort(key=lambda x: (x[0], 1 if x[1] == 'end' else 0))
    
    total_area = 0
    active_y = []
    prev_x = None
    
    for event in events:
        x, etype, y1, y2 = event
        if prev_x is not None and prev_x < x:
            total_y_span = calculate_total_y_span(active_y)
            area = total_y_span * (x - prev_x)
            total_area += area
            prev_x = x
        if etype == 'start':
            add_y_interval(active_y, y1, y2)
        elif etype == 'end':
            remove_y_interval(active_y, y1, y2)
    
    print(total_area)

def calculate_total_y_span(active_y):
    total = 0
    for y in active_y:
        total += y[1] - y[0]
    return total

def add_y_interval(active_y, y1, y2):
    merged = []
    current = (y1, y2)
    for y in active_y:
        if current[1] < y[0]:
            merged.append(current)
            current = y
        elif current[0] > y[1]:
            merged.append(y)
        else:
            current = (min(current[0], y[0]), max(current[1], y[1]))
    merged.append(current)
    active_y[:] = merged

def remove_y_interval(active_y, y1, y2):
    remaining = [y for y in active_y if y != (y1, y2)]
    # Merge remaining intervals
    if remaining:
        sorted_y = sorted(remaining, key=lambda x: x[0])
        merged = [sorted_y[0]]
        for y in sorted_y[1:]:
            if y[0] <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], y[1]))
            else:
                merged.append(y)
        active_y[:] = merged
    else:
        active_y.clear()

if __name__ == '__main__':
    main()