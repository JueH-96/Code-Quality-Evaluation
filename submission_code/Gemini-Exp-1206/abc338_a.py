s = input()
if len(s) == 0:
    print("No")
elif len(s) == 1:
    if 'A' <= s[0] <= 'Z':
        print("Yes")
    else:
        print("No")
elif 'A' <= s[0] <= 'Z':
    flag = True
    for i in range(1, len(s)):
        if not ('a' <= s[i] <= 'z'):
            flag = False
            break
    if flag:
        print("Yes")
    else:
        print("No")
else:
    print("No")