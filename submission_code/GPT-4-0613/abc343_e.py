V1, V2, V3 = map(int, input().split())
if V1 < 0 or V2 < 0 or V3 < 0 or V1 > 3*7**3 or V2 > 3*7**3 or V3 > 3*7**3:
    print("No")
else:
    a = int((V1 + 2*V2 + 3*V3)**(1/3))
    if a**3 != V1 + 2*V2 + 3*V3:
        print("No")
    else:
        b = int((V2 + 3*V3)**0.5)
        if b**2 != V2 + 3*V3:
            print("No")
        else:
            c = V3
            if a < b or b < c:
                print("No")
            else:
                print("Yes")
                print(0, 0, 0, a, a, a, a-b, a-b, a-b+c)