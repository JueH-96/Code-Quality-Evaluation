S = input().strip()

state = 'A'
for c in S:
    if state == 'A':
        if c == 'A':
            continue
        elif c == 'B':
            state = 'B'
        elif c == 'C':
            state = 'C'
        else:
            print('No')
            exit()
    elif state == 'B':
        if c == 'B':
            continue
        elif c == 'C':
            state = 'C'
        else:
            print('No')
            exit()
    elif state == 'C':
        if c == 'C':
            continue
        else:
            print('No')
            exit()
    else:
        print('No')
        exit()
print('Yes')