n = int(input())
H = list(map(int, input().split()))

stack = []
ans = [0]*n
for i in range(n):
    while stack and stack[-1][0] < H[i]:
        h, idx, val = stack.pop()
        ans[idx] += H[i] + val - h
    if stack:
        ans[i] += stack[-1][2] + 1
        stack.append((H[i], i, stack[-1][2] + 1))
    else:
        stack.append((H[i], i, 0))
print(' '.join(map(str, ans)))