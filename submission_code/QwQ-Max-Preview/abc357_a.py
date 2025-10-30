# Read input
n, m = map(int, input().split())
h_list = list(map(int, input().split()))

remaining = m
count = 0

for h in h_list:
    if remaining < h:
        if remaining > 0:
            remaining = 0
        break
    count += 1
    remaining -= h
    if remaining == 0:
        break

print(count)