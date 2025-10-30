n = int(input())
for i in range(n, 1000):
    x = i // 100
    y = (i % 100) // 10
    z = i % 10
    if x * y == z:
        print(i)
        break