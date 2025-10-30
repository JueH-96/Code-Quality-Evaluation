import sys

def solve():
    N = int(input())
    cards = [tuple(map(int, input().split())) + (i + 1,) for i in range(N)]
    cards.sort(key=lambda x: x[0])

    stack = []
    for a, c, i in cards:
        while stack and stack[-1][1] > c:
            stack.pop()
        if not stack or stack[-1][1] < c:
            stack.append((a, c, i))

    print(len(stack))
    print(*[i for _, _, i in stack])

solve()