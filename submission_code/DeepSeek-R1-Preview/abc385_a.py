a, b, c = map(int, input().split())
s = a + b + c

if a == b == c:
    print("Yes")
else:
    if s % 2 != 0:
        print("No")
    else:
        target = s // 2
        if a == target or b == target or c == target:
            print("Yes")
        else:
            print("No")