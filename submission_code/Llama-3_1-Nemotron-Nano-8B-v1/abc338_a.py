s = input().strip()
if len(s) == 1:
    print("Yes" if s[0].isupper() else "No")
else:
    print("Yes" if s[0].isupper() and s[1:].islower() else "No")