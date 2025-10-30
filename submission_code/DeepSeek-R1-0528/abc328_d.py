def main():
    s = input().strip()
    stack = []
    for c in s:
        if c == 'C' and len(stack) >= 2 and stack[-1] == 'B' and stack[-2] == 'A':
            stack.pop()
            stack.pop()
        else:
            stack.append(c)
    print(''.join(stack))

if __name__ == "__main__":
    main()