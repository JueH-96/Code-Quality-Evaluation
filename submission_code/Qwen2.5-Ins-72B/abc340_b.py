# YOUR CODE HERE
Q = int(input())
A = []
results = []

for _ in range(Q):
    query = list(map(int, input().split()))
    if query[0] == 1:
        A.append(query[1])
    elif query[0] == 2:
        k = query[1]
        results.append(A[-k])

for result in results:
    print(result)