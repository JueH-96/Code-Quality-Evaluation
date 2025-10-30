s, t = input().split()
len_s = len(s)
found = False

for w in range(1, len_s):  # w must be less than len(s)
    for c in range(1, w + 1):  # c can be from 1 to w inclusive
        candidate = []
        for i in range(0, len_s, w):
            chunk = s[i:i+w]
            if len(chunk) >= c:
                candidate.append(chunk[c-1])
        if ''.join(candidate) == t:
            found = True
            break
    if found:
        break

print("Yes" if found else "No")