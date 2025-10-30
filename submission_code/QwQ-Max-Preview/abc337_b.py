import re

s = input().strip()

# Check if the string matches the pattern A*B*C*
if re.fullmatch(r'^A*B*C*$', s):
    print("Yes")
else:
    print("No")