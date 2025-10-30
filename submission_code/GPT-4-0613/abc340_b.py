Q = int(input().strip())
A = []
for _ in range(Q):
    query = list(map(int, input().strip().split()))
    if query[0] == 1:
        A.append(query[1])
    else:
        print(A[-query[1]])