import sys

s = input().strip()
t = input().strip()

for i in range(1, min(len(s), len(t)) + 1):
    if s[i-1] != t[i-1]:
        print(i)
        sys.exit()

if len(s) != len(t):
    print(min(len(s), len(t)) + 1)
else:
    print(0)