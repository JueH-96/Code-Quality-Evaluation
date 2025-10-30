# YOUR CODE HERE
def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    Q = int(data[N+1])
    queries = [list(map(int, data[N+2+i*2:N+4+i*2])) for i in range(Q)]
    
    # Precompute the sleep intervals
    sleep_intervals = []
    for i in range(1, N, 2):
        sleep_intervals.append((A[i], A[i+1] - A[i]))
    
    # Precompute the prefix sums of sleep intervals
    prefix_sums = [0] * (N // 2 + 1)
    for i, (start, duration) in enumerate(sleep_intervals):
        prefix_sums[i+1] = prefix_sums[i] + duration
    
    def query(l, r):
        total_sleep = 0
        for i, (start, duration) in enumerate(sleep_intervals):
            if start >= l and start < r:
                end = min(start + duration, r)
                total_sleep += end - max(start, l)
        return total_sleep
    
    results = []
    for l, r in queries:
        results.append(query(l, r))
    
    for result in results:
        print(result)

solve()