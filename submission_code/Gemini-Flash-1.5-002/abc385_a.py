a, b, c = map(int, input().split())
total = a + b + c
if total % 2 == 0 and max(a, b, c) <= total // 2:
    print("Yes")
elif total % 3 == 0 and max(a, b, c) <= total // 3:
    print("Yes")
elif a == b == c:
    print("Yes")
else:
    print("No")