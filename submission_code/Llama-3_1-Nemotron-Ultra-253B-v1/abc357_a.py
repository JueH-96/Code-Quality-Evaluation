n, m = map(int, input().split())
h_list = list(map(int, input().split()))
count = 0
for h in h_list:
    if m >= h:
        m -= h
        count += 1
    else:
        m = 0
    if m == 0:
        break
print(count)