S = input()

from collections import Counter
c = Counter(S)
cnt = Counter(c.values())

ans = 'Yes'
for k, v in cnt.items():
    if v % 2 != 0:
        ans = 'No'
        break

print(ans)