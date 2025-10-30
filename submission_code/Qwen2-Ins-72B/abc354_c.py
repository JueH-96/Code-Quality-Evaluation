from operator import itemgetter

N = int(input())
AC = [tuple(map(int, input().split())) for _ in range(N)]
AC = sorted(AC, key=itemgetter(1))

stack = []
for a, c in AC:
    while stack and stack[-1][0] < a:
        stack.pop()
    stack.append((a, c))

print(len(stack))
print(' '.join(map(str, sorted([i + 1 for i, _ in enumerate(AC) if (AC[i][0], AC[i][1]) in stack]))))