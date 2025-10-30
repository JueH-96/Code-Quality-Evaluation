s = input()
if 'A' <= s[0] <= 'Z':
    if len(s) == 1:
        print("Yes")
    else:
        all_lower = True
        for i in range(1, len(s)):
            if not ('a' <= s[i] <= 'z'):
                all_lower = False
                break
        if all_lower:
            print("Yes")
        else:
            print("No")
else:
    print("No")