n, m = map(int, input().split())
h_list = list(map(int, input().split()))

current_m = m
count = 0

for h in h_list:
    if current_m == 0:
        break
    if h <= current_m:
        current_m -= h
        count += 1
    else:
        current_m = 0
        break

print(count)