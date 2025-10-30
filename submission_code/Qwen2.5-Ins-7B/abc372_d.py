# N = int(input())
# H = list(map(int, input().split()))

N = int(input())
H = list(map(int, input().split()))

stack = []
ans = [0] * N

for i in range(N):
    while stack and H[stack[-1]] < H[i]:
        ans[stack.pop()] = i - stack[-1] if stack else i + 1
    stack.append(i)

while stack:
    ans[stack.pop()] = N - stack[-1] if stack else N

print(*ans)