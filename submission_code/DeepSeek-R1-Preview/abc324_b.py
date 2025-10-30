n = int(input())
if n == 1:
    print("Yes")
else:
    while n % 2 == 0:
        n = n // 2
    while n % 3 == 0:
        n = n // 3
    print("Yes" if n == 1 else "No")