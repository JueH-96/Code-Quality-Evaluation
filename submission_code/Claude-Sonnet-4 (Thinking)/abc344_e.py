N = int(input())
A = list(map(int, input().split()))
Q = int(input())

for _ in range(Q):
    query = list(map(int, input().split()))
    if query[0] == 1:  # Insert y after x
        x, y = query[1], query[2]
        idx = A.index(x)
        A.insert(idx + 1, y)
    else:  # Remove x
        x = query[1]
        A.remove(x)

print(' '.join(map(str, A)))