n, m = map(int, input().split())
h_list = list(map(int, input().split()))
count = 0
for h in h_list:
    if m >= h:
        count += 1
        m -= h
    else:
        break
print(count)