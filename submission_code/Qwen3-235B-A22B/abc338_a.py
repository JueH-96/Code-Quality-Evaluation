s = input().strip()
if s[0].isupper() and s[1:].lower() == s[1:]:
    print("Yes")
else:
    print("No")