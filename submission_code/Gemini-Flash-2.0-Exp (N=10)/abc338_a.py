s = input()
if len(s) == 0:
    print("No")
elif s[0].isupper():
    all_lower = True
    for i in range(1, len(s)):
        if not s[i].islower():
            all_lower = False
            break
    if all_lower:
        print("Yes")
    else:
        print("No")
else:
    print("No")