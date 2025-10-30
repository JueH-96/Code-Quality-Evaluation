def solve():
    n = int(input())
    events = []
    for _ in range(n):
        t, v = map(int, input().split())
        events.append((t, v))
    
    current_water = 0
    last_time = 0
    
    for i in range(n):
        time, volume = events[i]
        
        if current_water > 0:
            time_diff = time - last_time
            current_water = max(0, current_water - time_diff)
        
        current_water += volume
        last_time = time
        
    print(current_water)

solve()