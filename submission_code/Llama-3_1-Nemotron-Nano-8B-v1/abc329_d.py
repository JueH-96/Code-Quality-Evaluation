n, m = map(int, input().split())
a_list = list(map(int, input().split()))

counts = [0] * (n + 1)
max_count = 0
best = 0

for a in a_list:
    counts[a] += 1
    if counts[a] > max_count:
        max_count = counts[a]
        best = a
    elif counts[a] == max_count:
        if a < best:
            best = a
    print(best)