n, m = map(int, input().split())
h_list = list(map(int, input().split()))

remaining = m
count = 0

for h in h_list:
    if remaining >= h:
        count += 1
        remaining -= h
    else:
        if remaining > 0:
            remaining = 0
        break  # No more disinfectant left after this

print(count)