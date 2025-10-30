def solve():
    n, a = map(int, input().split())
    t = list(map(int, input().split()))
    
    finish_time = 0
    for arrival_time in t:
        start_time = max(finish_time, arrival_time)
        finish_time = start_time + a
        print(finish_time)

solve()