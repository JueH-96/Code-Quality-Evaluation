import sys

def solve():
    N = int(input())
    H = list(map(int, input().split()))
    ans = [0] * N
    stack = []
    for i in range(N):
        while stack and H[stack[-1]] < H[i]:
            ans[stack.pop()] += 1
        if stack:
            ans[stack[-1]] += 1
        stack.append(i)
    print(' '.join(map(str, ans)))

solve()