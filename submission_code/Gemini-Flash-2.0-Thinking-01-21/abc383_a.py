def solve():
    n = int(input())
    events = []
    for _ in range(n):
        t_i, v_i = map(int, input().split())
        events.append({'time': t_i, 'volume': v_i})
    
    current_water_level = 0
    last_event_time = 0
    
    for event in events:
        event_time = event['time']
        volume_added = event['volume']
        
        time_elapsed = event_time - last_event_time
        if current_water_level > 0:
            water_lost = time_elapsed
            current_water_level = max(0, current_water_level - water_lost)
            
        current_water_level += volume_added
        last_event_time = event_time
        
    print(current_water_level)

if __name__ == '__main__':
    solve()