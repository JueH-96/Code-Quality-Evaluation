n = int(input())
for i in range(n, 1000):
    h = i // 100
    t = (i // 10) % 10
    o = i % 10
    if h * t == o:
        print(i)
        break