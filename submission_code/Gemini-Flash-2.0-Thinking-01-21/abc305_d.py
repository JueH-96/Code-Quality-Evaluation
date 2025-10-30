def solve():
    n = int(input())
    a = list(map(int, input().split()))
    q = int(input())
    queries = []
    for _ in range(q):
        queries.append(list(map(int, input().split())))
    
    sleep_sessions = []
    for i in range(1, n // 2 + 1):
        sleep_sessions.append((a[2*i-1], a[2*i]))
        
    results = []
    for l_i, r_i in queries:
        total_sleep_time = 0
        for sleep_start, sleep_end in sleep_sessions:
            intersection_start = max(sleep_start, l_i)
            intersection_end = min(sleep_end, r_i + 1)
            if intersection_start < intersection_end:
                total_sleep_time += (intersection_end - intersection_start)
        results.append(total_sleep_time)
        
    for result in results:
        print(result)

if __name__ == '__main__':
    solve()