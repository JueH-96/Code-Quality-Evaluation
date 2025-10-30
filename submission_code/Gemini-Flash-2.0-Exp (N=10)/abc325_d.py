def solve():
    n = int(input())
    intervals = []
    for _ in range(n):
        t, d = map(int, input().split())
        intervals.append((t, t + d))

    events = []
    for start, end in intervals:
        events.append((start, 1))  # 1 for start
        events.append((end, -1))  # -1 for end
    
    events.sort()
    
    max_printed = 0
    printed_count = 0
    last_print_time = -float('inf')
    
    
    active_intervals = []
    
    event_idx = 0
    while event_idx < len(events):
        current_time = events[event_idx][0]
        
        
        while event_idx < len(events) and events[event_idx][0] == current_time:
            if events[event_idx][1] == 1:
                active_intervals.append(events[event_idx][0])
            else:
                active_intervals.remove(events[event_idx][0] - (events[event_idx][0] - events[event_idx][0]))
            event_idx += 1
        
        if active_intervals and current_time >= last_print_time + 1:
            printed_count += 1
            last_print_time = current_time
            
    
    print(printed_count)

solve()