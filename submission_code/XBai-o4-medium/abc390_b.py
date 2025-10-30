n = int(input())
a = list(map(int, input().split()))
if n == 2:
    print("Yes")
else:
    r_num = a[1]
    r_den = a[0]
    valid = True
    for i in range(1, n):
        if a[i] * r_den != a[i-1] * r_num:
            valid = False
            break
    print("Yes" if valid else "No")