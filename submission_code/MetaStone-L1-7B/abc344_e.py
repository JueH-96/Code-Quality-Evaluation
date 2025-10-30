n = int(input())
A = list(map(int, input().split()))
q = int(input())
queries = [input().split() for _ in range(q)]

prev = {}
next = {}
head = None
tail = None

for i in range(len(A)):
    current = A[i]
    if i == 0:
        head = current
        tail = current
    else:
        prev[current] = A[i-1]
        next[A[i-1]] = current
        tail = current

for query in queries:
    parts = query
    if parts[0] == '1':
        x = int(parts[1])
        y = int(parts[2])
        prev_x = prev.get(x)
        next_x = next.get(x)
        prev[y] = x
        next[y] = next_x
        if next_x is not None:
            prev[next_x] = y
        next[x] = y
    else:
        x = int(parts[1])
        prev_x = prev.get(x)
        next_x = next.get(x)
        if prev_x is not None:
            next[prev_x] = next_x
        if next_x is not None:
            prev[next_x] = prev_x
        del prev[x]
        del next[x]
        if head == x:
            head = next_x
        if tail == x:
            tail = prev_x

result = []
current = head
while current is not None:
    result.append(str(current))
    current = next.get(current, None)

print(' '.join(result))