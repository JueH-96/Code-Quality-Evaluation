s = input().strip()

if not s:
    print("No")
else:
    if s[0].isupper():
        all_lower = True
        for c in s[1:]:
            if not c.islower():
                all_lower = False
                break
        print("Yes" if all_lower else "No")
    else:
        print("No")