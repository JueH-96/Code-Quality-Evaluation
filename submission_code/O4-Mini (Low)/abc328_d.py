import sys

def main():
    S = sys.stdin.readline().strip()
    stack = []
    for ch in S:
        stack.append(ch)
        if len(stack) >= 3 and stack[-3] == 'A' and stack[-2] == 'B' and stack[-1] == 'C':
            # remove the 'ABC' substring
            stack.pop()
            stack.pop()
            stack.pop()
    # join the remaining characters to form the final string
    print("".join(stack))

if __name__ == "__main__":
    main()