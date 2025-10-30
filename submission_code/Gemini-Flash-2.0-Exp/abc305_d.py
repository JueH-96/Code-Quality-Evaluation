def solve():
    n = int(input())
    a = list(map(int, input().split()))
    q = int(input())
    
    for _ in range(q):
        l, r = map(int, input().split())
        
        total_sleep_time = 0
        
        for i in range(1, (n - 1) // 2 + 1):
            sleep_start = a[2 * i - 1]
            sleep_end = a[2 * i]
            
            intersection_start = max(l, sleep_start)
            intersection_end = min(r, sleep_end)
            
            if intersection_start < intersection_end:
                total_sleep_time += (intersection_end - intersection_start)
        
        print(total_sleep_time)

solve()