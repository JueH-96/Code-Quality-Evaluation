# YOUR CODE HERE
def count_intersecting_intervals():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    intervals = []
    
    index = 1
    for _ in range(N):
        l = int(data[index])
        r = int(data[index + 1])
        intervals.append((l, r))
        index += 2
    
    events = []
    for l, r in intervals:
        events.append((l, 'start'))
        events.append((r, 'end'))
    
    # Sort events. If two events have the same position, 'start' should come before 'end'
    events.sort(key=lambda x: (x[0], x[1] == 'end'))
    
    active_intervals = 0
    intersection_count = 0
    
    for position, event_type in events:
        if event_type == 'start':
            # Count intersections with currently active intervals
            intersection_count += active_intervals
            # Add this interval to active intervals
            active_intervals += 1
        elif event_type == 'end':
            # Remove this interval from active intervals
            active_intervals -= 1
    
    print(intersection_count)