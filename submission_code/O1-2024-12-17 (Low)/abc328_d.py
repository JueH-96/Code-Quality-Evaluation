def main():
    import sys
    S = sys.stdin.readline().strip()

    stack = []
    for ch in S:
        stack.append(ch)
        # Whenever the last three characters form "ABC", remove them
        if len(stack) >= 3 and ''.join(stack[-3:]) == 'ABC':
            stack[-3:] = []

    print(''.join(stack))

# Do not forget to call main()!
if __name__ == "__main__":
    main()