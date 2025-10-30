def solve():
    n = int(input())
    a = list(map(int, input().split()))
    q = int(input())

    for _ in range(q):
        l, r = map(int, input().split())
        total_sleep_time = 0
        for i in range(1, n // 2 + 1):
            sleep_start = a[2 * i]
            sleep_end = a[2 * i + 1]
            overlap = max(0, min(r, sleep_end) - max(l, sleep_start))
            total_sleep_time += overlap
        print(total_sleep_time)

solve()