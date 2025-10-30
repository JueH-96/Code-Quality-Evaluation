import sys
from collections import deque

def solve():
    N = int(sys.stdin.readline().strip())
    H = list(map(int, sys.stdin.readline().strip().split()))
    stack = deque()
    result = [0] * N

    for i in range(N-1, -1, -1):
        while stack and H[i] >= H[stack[-1]]:
            stack.pop()
        result[i] = stack[-1] if stack else N
        stack.append(i)

    for i in range(N):
        print(result[i] - i - 1 if result[i] != N else 0, end=' ')

solve()