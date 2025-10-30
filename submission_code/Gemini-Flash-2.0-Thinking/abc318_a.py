n, m, p = map(int, input().split())

if m > n:
    print(0)
else:
    count = 0
    current_full_moon = m
    while current_full_moon <= n:
        count += 1
        current_full_moon += p
    print(count)