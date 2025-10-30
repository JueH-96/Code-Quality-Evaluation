def main():
    S = input().strip()
    stack = []
    for c in S:
        if c != 'A':
            stack.append(c)
        else:
            count = 0
            while stack and stack[-1] == 'W':
                stack.pop()
                count += 1
            if count > 0:
                stack.append('A')
                stack.extend(['C'] * count)
            else:
                stack.append('A')
    print(''.join(stack))

if __name__ == "__main__":
    main()