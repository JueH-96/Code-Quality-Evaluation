def solve():
    n, a = map(int, input().split())
    t = list(map(int, input().split()))
    
    finish_times = []
    current_time = 0
    
    for arrival_time in t:
        if current_time <= arrival_time:
            current_time = arrival_time + a
        else:
            current_time += a
        finish_times.append(current_time)
        
    for time in finish_times:
        print(time)

solve()