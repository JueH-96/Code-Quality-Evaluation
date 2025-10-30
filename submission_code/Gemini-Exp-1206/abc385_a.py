a, b, c = map(int, input().split())

if a + b == c or a + c == b or b + c == a:
    print("Yes")
elif a == b or a == c or b == c:
    print("Yes")
elif a == b and b == c:
    print("Yes")
else:
    print("No")