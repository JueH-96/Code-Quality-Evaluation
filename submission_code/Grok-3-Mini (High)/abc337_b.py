import sys
s = sys.stdin.readline().strip()
if all(s[i] <= s[i+1] for i in range(len(s) - 1)):
    print("Yes")
else:
    print("No")