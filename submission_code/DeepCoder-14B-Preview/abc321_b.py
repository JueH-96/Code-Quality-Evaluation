n, x = map(int, input().split())
a = list(map(int, input().split()))

for s in range(0, 101):
    temp = a + [s]
    temp.sort()
    total = sum(temp[1:-1])
    if total >= x:
        print(s)
        exit()

print(-1)