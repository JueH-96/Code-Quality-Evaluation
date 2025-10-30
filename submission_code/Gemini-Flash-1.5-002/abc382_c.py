n, m = map(int, input().split())
a = list(map(int, input().split()))
b = list(map(int, input().split()))

for sushi in b:
    eaten = False
    for i in range(n):
        if sushi >= a[i]:
            print(i + 1)
            eaten = True
            break
    if not eaten:
        print(-1)