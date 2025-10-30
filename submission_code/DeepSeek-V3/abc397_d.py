Q = int(input())
stack = [0] * 100  # Initial stack with 100 zeros
top = 99  # Points to the top of the stack (initially the last element since we add new elements by decreasing top)

output = []
for _ in range(Q):
    query = input().split()
    if query[0] == '1':
        x = int(query[1])
        stack[top] = x
        top -= 1
    else:
        top += 1
        output.append(stack[top])

print('
'.join(map(str, output)))