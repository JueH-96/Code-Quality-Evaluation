def solve():
    n = int(input())
    events = []
    for _ in range(n):
        t, v = map(int, input().split())
        events.append({'time': t, 'volume': v})
    
    current_water_level = 0
    last_event_time = 0
    
    for event in events:
        event_time = event['time']
        add_volume = event['volume']
        
        time_elapsed = event_time - last_event_time
        decrease_amount = min(current_water_level, time_elapsed)
        current_water_level = max(0, current_water_level - decrease_amount)
        current_water_level += add_volume
        last_event_time = event_time
        
    print(current_water_level)

if __name__ == '__main__':
    solve()