def is_326_like(n):
    s = str(n)
    return int(s[0]) * int(s[1]) == int(s[2])

n = int(input())
while True:
    if is_326_like(n):
        print(n)
        break
    n += 1