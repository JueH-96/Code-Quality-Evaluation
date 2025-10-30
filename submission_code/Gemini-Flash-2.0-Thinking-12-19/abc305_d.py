def solve():
    n = int(input())
    a = list(map(int, input().split()))
    q = int(input())
    queries = []
    for _ in range(q):
        l, r = map(int, input().split())
        queries.append({'l': l, 'r': r})
    
    sleep_intervals = []
    for i in range(1, n, 2):
        sleep_intervals.append({'start': a[i], 'end': a[i+1]})
        
    results = []
    for query in queries:
        l_query = query['l']
        r_query = query['r']
        total_sleep_time = 0
        for sleep_interval in sleep_intervals:
            start_sleep = sleep_interval['start']
            end_sleep = sleep_interval['end']
            intersection_start = max(start_sleep, l_query)
            intersection_end = min(end_sleep, r_query + 1)
            sleep_duration = max(0, intersection_end - intersection_start)
            total_sleep_time += sleep_duration
        results.append(total_sleep_time)
        
    for result in results:
        print(result)

if __name__ == '__main__':
    solve()