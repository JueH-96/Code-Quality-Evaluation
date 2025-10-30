n = int(input())
a = list(map(int, input().split()))

if n == 2:
    print("Yes")
else:
    a0 = a[0]
    a1 = a[1]
    ok = True
    for i in range(2, n):
        if a[i] * a0 != a[i-1] * a1:
            ok = False
            break
    print("Yes" if ok else "No")