A, B = map(int, input().split())

if A == 1:
    if B == 2 or B == 4:
        print("Yes")
    else:
        print("No")
elif A == 2:
    if B == 1 or B == 3 or B == 5:
        print("Yes")
    else:
        print("No")
elif A == 3:
    if B == 2 or B == 6:
        print("Yes")
    else:
        print("No")
elif A == 4:
    if B == 5 or B == 7:
        print("Yes")
    else:
        print("No")
elif A == 5:
    if B == 4 or B == 6 or B == 8:
        print("Yes")
    else:
        print("No")
elif A == 6:
    if B == 5 or B == 9:
        print("Yes")
    else:
        print("No")
elif A == 7:
    if B == 8:
        print("Yes")
    else:
        print("No")
elif A == 8:
    if B == 7 or B == 9:
        print("Yes")
    else:
        print("No")
else:
    print("No")