n, a = map(int, input().split())
times = list(map(int, input().split()))
current_finish = 0
for t in times:
    start = max(t, current_finish)
    current_finish = start + a
    print(current_finish)