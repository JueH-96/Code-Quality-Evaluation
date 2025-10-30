import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:N+1]))
Q = int(data[N+1])
queries = []
for i in range(Q):
    l = int(data[N+2+i*2])
    r = int(data[N+2+i*2+1])
    queries.append((l, r))

sleep_intervals = []
for i in range(1, (N-1)//2 + 1):
    sleep_intervals.append((A[2*i], A[2*i+1]))

def calculate_sleep_time(l, r, sleep_intervals):
    total_sleep = 0
    for sleep_start, sleep_end in sleep_intervals:
        if sleep_end < l or sleep_start > r:
            continue
        total_sleep += min(r, sleep_end) - max(l, sleep_start)
    return total_sleep

results = []
for l, r in queries:
    results.append(calculate_sleep_time(l, r, sleep_intervals))

sys.stdout.write("
".join(map(str, results)) + "
")