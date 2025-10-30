s = input()

if not s[0].isupper():
    print("No")
else:
    all_others_lower = True
    for i in range(1, len(s)):
        if not s[i].islower():
            all_others_lower = False
            break
    if all_others_lower:
        print("Yes")
    else:
        print("No")