A, B, C = map(int, input().split())
total = A + B + C

if total % 2 == 0:
    target = total // 2
    if A == target or B == target or C == target:
        print("Yes")
    elif (A + B == target) or (A + C == target) or (B + C == target):
        print("Yes")
    else:
        if total % 3 == 0:
            t = total // 3
            if A == t and B == t and C == t:
                print("Yes")
            else:
                print("No")
        else:
            print("No")
else:
    if total % 3 == 0:
        t = total // 3
        if A == t and B == t and C == t:
            print("Yes")
        else:
            print("No")
    else:
        print("No")