n, t, p = map(int, input().split())
l = list(map(int, input().split()))
initial = sum(1 for x in l if x >= t)
if initial >= p:
    print(0)
else:
    d = 1
    while True:
        current = sum(1 for x in l if x + d >= t)
        if current >= p:
            print(d)
            break
        d += 1