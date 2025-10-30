import sys

def main() -> None:
    S = sys.stdin.readline().strip()

    stack = []                # keeps the current string after deletions
    for ch in S:
        stack.append(ch)
        # if the three last characters form 'ABC', delete them
        if len(stack) >= 3 and stack[-3:] == ['A', 'B', 'C']:
            stack.pop()
            stack.pop()
            stack.pop()

    sys.stdout.write(''.join(stack) + '
')

if __name__ == "__main__":
    main()