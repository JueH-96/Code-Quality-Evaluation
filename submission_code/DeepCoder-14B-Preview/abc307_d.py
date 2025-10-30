n = int(input())
s = list(input())

while True:
    removed = [False] * len(s)
    stack = []
    prefix_paren = [0] * (len(s) + 1)
    
    for i in range(len(s)):
        prefix_paren[i+1] = prefix_paren[i] + (1 if s[i] in '()' else 0)
    
    for i in range(len(s)):
        if s[i] == '(':
            stack.append(i)
        elif s[i] == ')':
            if stack:
                j = stack.pop()
                # Check if there are no parentheses between j+1 and i-1
                count = prefix_paren[i] - prefix_paren[j+1]
                if count == 0:
                    for k in range(j, i+1):
                        if not removed[k]:
                            removed[k] = True
    
    new_s = [s[i] for i in range(len(s)) if not removed[i]]
    if len(new_s) == len(s):
        break
    s = new_s

print(''.join(s))