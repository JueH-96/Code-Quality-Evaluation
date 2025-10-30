n, m = map(int, input().split())
h = list(map(int, input().split()))

remaining = m
count = 0

for hands in h:
    if remaining >= hands:
        remaining -= hands
        count += 1
    else:
        remaining = 0

print(count)