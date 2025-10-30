def solve():
    n, a = map(int, input().split())
    t = list(map(int, input().split()))

    finish_time_previous = 0
    finish_times = []

    for arrival_time in t:
        start_time = max(arrival_time, finish_time_previous)
        finish_time = start_time + a
        finish_times.append(finish_time)
        finish_time_previous = finish_time

    for ft in finish_times:
        print(ft)

if __name__ == "__main__":
    solve()