l, r = map(int, input().split())

if l + r == 1:
    if l == 1:
        print("Yes")
    else:
        print("No")
else:
    print("Invalid")