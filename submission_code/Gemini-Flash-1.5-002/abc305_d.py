# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))
q = int(input())

for _ in range(q):
    l, r = map(int, input().split())
    total_sleep = 0
    for i in range(0, n - 1, 2):
        sleep_start = a[i + 1]
        sleep_end = a[i + 2]
        start = max(l, sleep_start)
        end = min(r, sleep_end)
        if start < end:
            total_sleep += end - start
    print(total_sleep)