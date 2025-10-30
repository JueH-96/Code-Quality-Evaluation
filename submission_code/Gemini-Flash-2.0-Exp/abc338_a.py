S = input()
if len(S) == 0:
    print("No")
else:
    if S[0].isupper():
        flag = True
        for i in range(1, len(S)):
            if not S[i].islower():
                flag = False
                break
        if flag:
            print("Yes")
        else:
            print("No")
    else:
        print("No")