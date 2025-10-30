n = int(input())
s = input().strip()

found = set()
for i in range(n):
    found.add(s[i])
    if len(found) == 3:
        print(i + 1)
        break