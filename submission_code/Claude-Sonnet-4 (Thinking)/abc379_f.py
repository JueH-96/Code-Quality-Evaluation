N, Q = map(int, input().split())
H = [0] + list(map(int, input().split()))  # 1-indexed

leftmost = [1] * (N + 1)
stack = []

for j in range(1, N + 1):
    while stack and H[stack[-1]] <= H[j]:
        stack.pop()
    if stack:
        leftmost[j] = stack[-1] + 1
    else:
        leftmost[j] = 1
    stack.append(j)

for _ in range(Q):
    l, r = map(int, input().split())
    count = 0
    for j in range(r + 1, N + 1):
        if leftmost[j] <= l:
            count += 1
    print(count)