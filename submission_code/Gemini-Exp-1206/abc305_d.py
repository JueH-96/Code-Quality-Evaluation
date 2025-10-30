def solve():
    n = int(input())
    a = list(map(int, input().split()))
    q = int(input())
    
    sleep_times = []
    for i in range(1, (n - 1) // 2 + 1):
        sleep_times.append((a[2*i-1], a[2*i]))

    for _ in range(q):
        l, r = map(int, input().split())
        total_sleep = 0
        for start, end in sleep_times:
            sleep_start = max(l, start)
            sleep_end = min(r, end)
            if sleep_start < sleep_end:
                total_sleep += sleep_end - sleep_start
        print(total_sleep)

solve()