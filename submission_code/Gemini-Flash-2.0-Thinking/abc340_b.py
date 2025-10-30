Q = int(input())
A = []
output = []
for _ in range(Q):
    query = input().split()
    query_type = int(query[0])
    if query_type == 1:
        x = int(query[1])
        A.append(x)
    elif query_type == 2:
        k = int(query[1])
        index = len(A) - k
        output.append(A[index])

for res in output:
    print(res)