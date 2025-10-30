SAB, SAC, SBC = input().split()

if SAB == "<":
    if SAC == "<":
        print("B")
    else:
        if SBC == "<":
            print("C")
        else:
            print("A")
else:
    if SAC == "<":
        if SBC == "<":
            print("A")
        else:
            print("C")
    else:
        print("B")