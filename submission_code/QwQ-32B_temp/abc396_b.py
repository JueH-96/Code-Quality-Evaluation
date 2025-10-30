stack = [0] * 100
Q = int(input())
for _ in range(Q):
    parts = input().split()
    if parts[0] == '1':
        x = int(parts[1])
        stack.append(x)
    else:
        print(stack.pop())