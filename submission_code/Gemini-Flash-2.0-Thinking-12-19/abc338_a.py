s = input()
if len(s) == 1:
    if s[0].isupper():
        print("Yes")
    else:
        print("No")
else:
    if s[0].isupper():
        is_rest_lower = True
        for i in range(1, len(s)):
            if not s[i].islower():
                is_rest_lower = False
                break
        if is_rest_lower:
            print("Yes")
        else:
            print("No")
    else:
        print("No")