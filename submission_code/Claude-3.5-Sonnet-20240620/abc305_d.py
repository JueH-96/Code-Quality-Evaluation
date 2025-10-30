# YOUR CODE HERE
def calculate_sleep_time(sleep_log, start, end):
    total_sleep = 0
    for i in range(1, len(sleep_log) - 1, 2):
        sleep_start = max(sleep_log[i], start)
        sleep_end = min(sleep_log[i+1], end)
        if sleep_start < sleep_end:
            total_sleep += sleep_end - sleep_start
    return total_sleep

N = int(input())
sleep_log = list(map(int, input().split()))

Q = int(input())
for _ in range(Q):
    l, r = map(int, input().split())
    print(calculate_sleep_time(sleep_log, l, r))