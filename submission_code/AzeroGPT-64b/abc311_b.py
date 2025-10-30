n, d = map(int, input().split())
s_list = [input() for _ in range(n)]
counter_list = []
for i in range(d):
    counter = 0
    for j in range(n):
        if s_list[j][i] == 'o':
            counter += 1
    counter_list.append(counter)
count = 0
count_max = 0
for i in range(d):
    if counter_list[i] == n:
        count += 1
    else:
        count_max = max(count, count_max)
        count = 0
count_max = max(count, count_max)
print(count_max)