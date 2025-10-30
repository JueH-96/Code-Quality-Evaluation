# YOUR CODE HERE
w, b = map(int, input().split())
s = 'wbwbwwbwbwbw'
n = len(s)

if b == 0:
    if 'w' * w in s:
        print('Yes')
    else:
        print('No')
elif w == 0:
    if 'b' * b in s:
        print('Yes')
    else:
        print('No')

else:
    found = False
    for i in range(n):
        for j in range(i, 2 * n):
            sub = ''
            for k in range(i, j + 1):
                sub += s[k % n]
            
            wc = sub.count('w')
            bc = sub.count('b')
            
            if wc == w and bc == b:
                found = True
                break
        if found:
            break
    
    if found:
        print('Yes')
    else:
        print('No')