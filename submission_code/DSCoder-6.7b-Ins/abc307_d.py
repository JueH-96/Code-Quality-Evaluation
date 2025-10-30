import sys

N = int(sys.stdin.readline().strip())
S = sys.stdin.readline().strip()

def solve(S):
    stack = []
    for c in S:
        if c == ')':
            tmp = []
            while stack and stack[-1] != '(':
                tmp.append(stack.pop())
            if stack:
                stack.pop()
            while tmp:
                stack.append(tmp.pop())
        else:
            stack.append(c)
    return ''.join(stack)

print(solve(S))