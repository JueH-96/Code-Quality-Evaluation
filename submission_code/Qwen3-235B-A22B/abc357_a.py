n, m = map(int, input().split())
h_list = list(map(int, input().split()))

count = 0
remaining = m

for h in h_list:
    if remaining >= h:
        count += 1
        remaining -= h
    else:
        if remaining > 0:
            remaining = 0
        break

print(count)