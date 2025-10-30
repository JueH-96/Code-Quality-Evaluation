def solve():
    n, a = map(int, input().split())
    t = list(map(int, input().split()))

    finish_times = []
    last_finish_time = 0

    for i in range(n):
        arrival_time = t[i]
        start_time = max(arrival_time, last_finish_time)
        finish_time = start_time + a
        finish_times.append(finish_time)
        last_finish_time = finish_time

    for finish_time in finish_times:
        print(finish_time)

if __name__ == "__main__":
    solve()