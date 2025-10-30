S = input().strip()

state = 'A'  # Initial state

for char in S:
    if state == 'A':
        if char == 'A':
            continue
        elif char == 'B':
            state = 'B'
        elif char == 'C':
            state = 'C'
    elif state == 'B':
        if char == 'B':
            continue
        elif char == 'C':
            state = 'C'
        else:
            print("No")
            break
    elif state == 'C':
        if char == 'C':
            continue
        else:
            print("No")
            break
else:
    print("Yes")