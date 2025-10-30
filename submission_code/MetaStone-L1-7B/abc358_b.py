n, a = map(int, input().split())
t_list = list(map(int, input().split()))
prev_finish = 0
for t in t_list:
    start = max(t, prev_finish)
    finish = start + a
    print(finish)
    prev_finish = finish