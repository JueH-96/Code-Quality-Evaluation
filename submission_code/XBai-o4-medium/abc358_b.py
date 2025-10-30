n, a = map(int, input().split())
t_list = list(map(int, input().split()))
current_time = 0
for t in t_list:
    start = max(current_time, t)
    finish = start + a
    print(finish)
    current_time = finish