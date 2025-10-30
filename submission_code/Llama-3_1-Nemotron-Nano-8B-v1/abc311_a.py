n = int(input())
s = input().strip()

first = {'A': None, 'B': None, 'C': None}

for i in range(len(s)):
    c = s[i]
    if first[c] is None:
        first[c] = i + 1
    if all(v is not None for v in first.values()):
        print(max(first.values()))
        break