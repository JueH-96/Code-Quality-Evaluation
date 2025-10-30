s, t = input().split()

found = False
for w in range(1, len(s)):
    splits = [s[i:i+w] for i in range(0, len(s), w)]
    for c in range(1, w + 1):
        constructed = []
        for substr in splits:
            if len(substr) >= c:
                constructed.append(substr[c-1])
        if ''.join(constructed) == t:
            found = True
            break
    if found:
        break

print("Yes" if found else "No")