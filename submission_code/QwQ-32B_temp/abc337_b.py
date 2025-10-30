s = input().strip()
phase = 'A'
valid = True

for c in s:
    if phase == 'A':
        if c == 'A':
            continue
        elif c == 'B':
            phase = 'B'
        else:  # must be 'C'
            phase = 'C'
    elif phase == 'B':
        if c == 'B':
            continue
        elif c == 'C':
            phase = 'C'
        else:
            valid = False
            break
    elif phase == 'C':
        if c != 'C':
            valid = False
            break
    else:
        valid = False
        break

print("Yes" if valid else "No")