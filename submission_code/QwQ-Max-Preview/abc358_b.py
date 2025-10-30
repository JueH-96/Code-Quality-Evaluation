n, a = map(int, input().split())
times = list(map(int, input().split()))
prev_finish = 0
for t in times:
    start = max(t, prev_finish)
    finish = start + a
    print(finish)
    prev_finish = finish