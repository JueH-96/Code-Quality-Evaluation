Q = int(input())
stack = [0] * 100
top = 99
outputs = []

for _ in range(Q):
    query = input().split()
    if query[0] == '1':
        x = int(query[1])
        top += 1
        stack[top] = x
    else:
        outputs.append(stack[top])
        top -= 1

for output in outputs:
    print(output)