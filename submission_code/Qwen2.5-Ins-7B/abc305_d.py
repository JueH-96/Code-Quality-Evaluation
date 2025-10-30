# N = int(input())
# A = list(map(int, input().split()))
# Q = int(input())
# queries = [tuple(map(int, input().split())) for _ in range(Q)]

N = int(input())
A = list(map(int, input().split()))
Q = int(input())
queries = [tuple(map(int, input().split())) for _ in range(Q)]

sleep_intervals = [(A[i], A[i + 1]) for i in range(1, N, 2)]

def count_sleep_minutes(start, end, intervals):
    count = 0
    for interval in intervals:
        if interval[0] > end or interval[1] < start:
            continue
        if interval[0] >= start and interval[1] <= end:
            count += interval[1] - interval[0]
        elif interval[0] < start and interval[1] > start:
            count += min(end, interval[1]) - start
        elif interval[0] > end and interval[1] > end:
            count += end - max(start, interval[0])
    return count

for l, r in queries:
    print(count_sleep_minutes(l, r, sleep_intervals))