s = input()

if sum(1 for c in s if c.isupper()) > sum(1 for c in s if c.islower()):
    print(s.upper())
else:
    print(s.lower())