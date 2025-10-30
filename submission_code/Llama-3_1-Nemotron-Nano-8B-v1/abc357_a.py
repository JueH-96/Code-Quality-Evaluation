n, m = map(int, input().split())
h_list = list(map(int, input().split()))
remaining = m
count = 0
for h in h_list:
    if remaining >= h:
        remaining -= h
        count += 1
    else:
        remaining = 0
print(count)