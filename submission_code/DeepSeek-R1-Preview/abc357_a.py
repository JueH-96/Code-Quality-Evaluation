n, m = map(int, input().split())
h = list(map(int, input().split()))
count = 0
for hands in h:
    if m < hands:
        m = 0
        break
    m -= hands
    count += 1
print(count)