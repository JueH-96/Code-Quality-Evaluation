def main():
    S = input().strip()
    stack = []
    for c in S:
        stack.append(c)
        if len(stack) >= 3 and stack[-3] == 'A' and stack[-2] == 'B' and stack[-1] == 'C':
            stack.pop()
            stack.pop()
            stack.pop()
    print(''.join(stack))

if __name__ == '__main__':
    main()