s = input().strip()
if s[0].isupper() and (len(s) == 1 or s[1:].islower()):
    print("Yes")
else:
    print("No")