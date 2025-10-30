Q = int(input())
A = []

for _ in range(Q):
    query = input().strip()
    parts = query.split()
    if parts[0] == '1':
        x = int(parts[1])
        A.append(x)
    else:
        k = int(parts[1])
        print(A[-k])