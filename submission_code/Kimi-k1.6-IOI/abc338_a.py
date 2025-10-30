s = input().strip()
if len(s) == 0:
    print("No")
else:
    first_upper = s[0].isupper()
    rest_lower = all(c.islower() for c in s[1:])
    print("Yes" if first_upper and rest_lower else "No")