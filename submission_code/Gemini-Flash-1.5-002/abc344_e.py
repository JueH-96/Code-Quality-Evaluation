# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))
q = int(input())

for _ in range(q):
    query = list(map(int, input().split()))
    if query[0] == 1:
        x = query[1]
        y = query[2]
        try:
            index = a.index(x)
            a.insert(index + 1, y)
        except ValueError:
            pass
    else:
        x = query[1]
        try:
            a.remove(x)
        except ValueError:
            pass

print(*a)