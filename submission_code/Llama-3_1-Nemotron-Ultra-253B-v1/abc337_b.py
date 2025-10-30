import re

s = input().strip()
if re.fullmatch(r'A*B*C*', s):
    print("Yes")
else:
    print("No")