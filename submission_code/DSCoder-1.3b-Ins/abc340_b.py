Q = int(input())
A = []

for _ in range(Q):
    query = input().split()
    if query[0] == '1':
        A.append(int(query[1]))
    elif query[0] == '2':
        k = int(query[1])
        print(A[-k])