n, c = map(int, input().split())
t_list = list(map(int, input().split()))

if n == 0:
    print(0)
else:
    count = 1
    last = t_list[0]
    for t in t_list[1:]:
        if t - last >= c:
            count += 1
            last = t
    print(count)