def main():
    n = int(input().strip())
    s = input().strip()
    n = len(s)
    
    matched = [False] * n
    stack = []
    for i, c in enumerate(s):
        if c == '(':
            stack.append(i)
        elif c == ')':
            if stack:
                j = stack.pop()
                matched[j] = True
                matched[i] = True
                
    depth = 0
    result_chars = []
    for i, c in enumerate(s):
        if c == '(' and matched[i]:
            depth += 1
        elif c == ')' and matched[i]:
            depth -= 1
        else:
            if depth == 0:
                result_chars.append(c)
                
    print(''.join(result_chars))

if __name__ == '__main__':
    main()