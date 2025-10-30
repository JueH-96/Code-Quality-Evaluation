n = int(input())
for i in range(n, 1000):
    a = i // 100
    b = (i // 10) % 10
    c = i % 10
    if a * b == c:
        print(i)
        break