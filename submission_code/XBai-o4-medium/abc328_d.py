def main():
    import sys
    S = sys.stdin.readline().strip()
    stack = []
    for c in S:
        stack.append(c)
        while len(stack) >= 3 and stack[-3:] == ['A', 'B', 'C']:
            # Remove the last three elements
            stack.pop()
            stack.pop()
            stack.pop()
    print(''.join(stack))

if __name__ == "__main__":
    main()