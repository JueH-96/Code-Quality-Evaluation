def next_326_like_number(n):
    while True:
        n += 1
        if n % 10 == 0:
            continue
        if n % 10 * (n // 100) % 10 == (n // 10) % 10:
            return n

n = int(input())
print(next_326_like_number(n))