n, a = map(int, input().split())
times = list(map(int, input().split()))
current_end = 0
for t in times:
    start = max(t, current_end)
    finish = start + a
    print(finish)
    current_end = finish