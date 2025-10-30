# YOUR CODE HERE
def count_covered_area():
    N = int(input())
    sheets = []
    for _ in range(N):
        sheets.append(list(map(int, input().split())))

    events = []
    for a, b, c, d in sheets:
        events.append((a, 0, c, d))  # Start of rectangle
        events.append((b, 1, c, d))  # End of rectangle

    events.sort()

    active = []
    prev_x = events[0][0]
    total_area = 0

    for x, event_type, y1, y2 in events:
        if x > prev_x:
            y_ranges = sorted(active)
            covered = 0
            prev_y = 0
            for start, end in y_ranges:
                start = max(start, prev_y)
                if start < end:
                    covered += end - start
                    prev_y = end
            total_area += covered * (x - prev_x)
        
        if event_type == 0:  # Start of rectangle
            active.append((y1, y2))
        else:  # End of rectangle
            active.remove((y1, y2))
        
        prev_x = x

    print(total_area)

count_covered_area()