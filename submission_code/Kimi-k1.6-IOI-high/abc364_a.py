import sys

n = int(sys.stdin.readline())
s = [sys.stdin.readline().strip() for _ in range(n)]

found = False
for i in range(n-1):
    if s[i] == 'sweet' and s[i+1] == 'sweet':
        if (i+1) != (n-1):
            found = True
            break

print("No" if found else "Yes")