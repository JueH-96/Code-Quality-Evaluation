def solve():
    n = int(input())
    a = list(map(int, input().split()))
    q = int(input())

    for _ in range(q):
        l, r = map(int, input().split())
        total_sleep_time = 0
        for i in range(1, n, 2):
            sleep_start = a[i]
            sleep_end = a[i+1]
            
            overlap_start = max(l, sleep_start)
            overlap_end = min(r, sleep_end)
            
            if overlap_end > overlap_start:
                total_sleep_time += (overlap_end - overlap_start)
        print(total_sleep_time)

solve()