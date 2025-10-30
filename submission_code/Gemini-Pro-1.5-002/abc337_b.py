# YOUR CODE HERE
s = input()
n = len(s)
state = 'A'
i = 0
while i < n:
    if state == 'A':
        if s[i] == 'A':
            i += 1
        elif s[i] == 'B':
            state = 'B'
            i += 1
        else:
            print('No')
            break
    elif state == 'B':
        if s[i] == 'B':
            i += 1
        elif s[i] == 'C':
            state = 'C'
            i += 1
        else:
            print('No')
            break
    elif state == 'C':
        if s[i] == 'C':
            i += 1
        else:
            print('No')
            break
else:
    print('Yes')