n = int(input())
for x in range(n, 1000):
    a = x // 100
    b = (x // 10) % 10
    c = x % 10
    if a * b == c:
        print(x)
        break