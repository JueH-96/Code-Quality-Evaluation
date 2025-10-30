n, a = map(int, input().split())
t_list = list(map(int, input().split()))

current_end = 0
for t in t_list:
    if current_end <= t:
        finish = t + a
    else:
        finish = current_end + a
    print(finish)
    current_end = finish