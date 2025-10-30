n, a = map(int, input().split())
times = list(map(int, input().split()))
prev_finish = 0
for t in times:
    current = max(t, prev_finish) + a
    print(current)
    prev_finish = current