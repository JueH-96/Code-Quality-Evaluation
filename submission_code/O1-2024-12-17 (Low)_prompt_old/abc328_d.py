def solve():
    import sys

    S = sys.stdin.readline().strip()
    stack = []

    for ch in S:
        stack.append(ch)
        # Whenever the last 3 characters are 'ABC', pop them
        if len(stack) >= 3 and ''.join(stack[-3:]) == 'ABC':
            stack.pop()
            stack.pop()
            stack.pop()

    print(''.join(stack))

def main():
    solve()

if __name__ == "__main__":
    main()