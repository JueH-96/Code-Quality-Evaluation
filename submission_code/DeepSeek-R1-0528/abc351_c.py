n = int(input())
A = list(map(int, input().split()))
stack = []
for a in A:
    while stack and stack[-1] == a:
        stack.pop()
        a += 1
    stack.append(a)
print(len(stack))