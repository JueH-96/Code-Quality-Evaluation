n, a = map(int, input().split())
t = list(map(int, input().split()))
for i in range(n):
    if i == 0:
        print(t[i] + a)
    else:
        if t[i] < t[i-1] + a:
            print(t[i-1] + a)
        else:
            print(t[i] + a)