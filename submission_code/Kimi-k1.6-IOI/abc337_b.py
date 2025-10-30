import re

s = input().strip()
print("Yes" if re.fullmatch(r'A*B*C*', s) else "No")