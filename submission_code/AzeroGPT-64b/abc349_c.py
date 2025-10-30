s = input()
t = input()

found = False

for i in range(len(s)):
    if s[i].upper() == t[0]:
        for j in range(i, len(s)):
            if s[j].upper() == t[1]:
                if s.upper().find(t[2], j) != -1:
                    found = True
                    break
        if found:
            break

if not found:
    for i in range(len(s)):
        if s[i].upper() == t[0]:
            for j in range(i, len(s)):
                if s[j].upper() == t[1] and s.upper().find('X', j) != -1:
                    found = True
                    break
            if found:
                break

if found:
    print("Yes")
else:
    print("No")