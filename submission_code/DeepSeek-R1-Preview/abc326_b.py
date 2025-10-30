n = int(input())
x = n
while True:
    a = x // 100
    b = (x // 10) % 10
    c = x % 10
    if a * b == c:
        print(x)
        break
    x += 1