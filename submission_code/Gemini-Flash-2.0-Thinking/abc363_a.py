r = int(input())

if 1 <= r <= 99:
    increase = 100 - r
elif 100 <= r <= 199:
    increase = 200 - r
elif 200 <= r <= 299:
    increase = 300 - r

print(increase)