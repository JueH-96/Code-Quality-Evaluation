n, a = map(int, input().split())
times = list(map(int, input().split()))
prev_end = 0
for t in times:
    start = max(t, prev_end)
    end = start + a
    print(end)
    prev_end = end