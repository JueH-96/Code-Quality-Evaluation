Q = int(input())
A = []
output = []
for _ in range(Q):
    query = list(map(int, input().split()))
    if query[0] == 1:
        A.append(query[1])
    elif query[0] == 2:
        k = query[1]
        output.append(A[-k])
for item in output:
    print(item)