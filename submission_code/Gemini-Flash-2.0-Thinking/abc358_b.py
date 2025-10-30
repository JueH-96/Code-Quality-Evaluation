def solve():
    n, a = map(int, input().split())
    t = list(map(int, input().split()))

    finish_times = []
    previous_finish_time = 0

    for arrival_time in t:
        start_time = max(arrival_time, previous_finish_time)
        finish_time = start_time + a
        finish_times.append(finish_time)
        previous_finish_time = finish_time

    for finish_time in finish_times:
        print(finish_time)

solve()