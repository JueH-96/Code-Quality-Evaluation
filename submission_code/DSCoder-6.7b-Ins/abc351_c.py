import sys
from collections import deque

def solve():
    N = int(sys.stdin.readline().strip())
    A = list(map(int, sys.stdin.readline().strip().split()))

    A = [2**a for a in A]
    stack = deque()

    for a in A:
        if not stack:
            stack.append(a)
        else:
            if stack[-1] == a:
                stack.pop()
                stack.append(a*2)
            else:
                stack.append(a)

    print(len(stack))

solve()