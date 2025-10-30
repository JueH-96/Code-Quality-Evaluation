n, m = map(int, input().split())
h_list = list(map(int, input().split()))
count = 0
remaining = m
for h in h_list:
    if remaining == 0:
        break
    if h <= remaining:
        count += 1
        remaining -= h
    else:
        remaining = 0
print(count)