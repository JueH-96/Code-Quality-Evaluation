# YOUR CODE HERE
S = input().strip()

# Track which phase we're in
# 0: can still see A's
# 1: have seen at least one B, can only see B's or C's
# 2: have seen at least one C, can only see C's
phase = 0

valid = True

for char in S:
    if phase == 0:  # A phase
        if char == 'A':
            continue
        elif char == 'B':
            phase = 1
        elif char == 'C':
            phase = 2
    elif phase == 1:  # B phase
        if char == 'A':
            valid = False
            break
        elif char == 'B':
            continue
        elif char == 'C':
            phase = 2
    elif phase == 2:  # C phase
        if char == 'A' or char == 'B':
            valid = False
            break
        elif char == 'C':
            continue

if valid:
    print("Yes")
else:
    print("No")