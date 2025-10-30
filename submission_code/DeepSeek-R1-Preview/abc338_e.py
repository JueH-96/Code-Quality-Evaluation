n = int(input())
events = []

for _ in range(n):
    a, b = map(int, input().split())
    if a > b:
        a, b = b, a
    events.append((a, 'start', b))
    events.append((b, 'end'))

# Sort the events by the point
events.sort(key=lambda x: x[0])

stack = []
for event in events:
    point, typ = event[0], event[1]
    if typ == 'start':
        stack.append(event[2])
    else:
        if not stack:
            print("Yes")
            exit()
        if stack[-1] != point:
            print("Yes")
            exit()
        stack.pop()

print("No")