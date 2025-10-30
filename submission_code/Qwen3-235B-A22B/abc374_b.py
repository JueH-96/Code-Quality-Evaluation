s = input().strip()
t = input().strip()

found = False
for i in range(min(len(s), len(t))):
    if s[i] != t[i]:
        print(i + 1)
        found = True
        break

if not found:
    if len(s) != len(t):
        print(min(len(s), len(t)) + 1)
    else:
        print(0)