s = input().strip()

if s[0].isupper() and all(c.islower() for c in s[1:]):
    print("Yes")
else:
    print("No")