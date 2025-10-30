n, a = map(int, input().split())
t_list = list(map(int, input().split()))
current_finish = 0
for t in t_list:
    start = max(t, current_finish)
    finish = start + a
    print(finish)
    current_finish = finish