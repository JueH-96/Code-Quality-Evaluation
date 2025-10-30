# YOUR CODE HERE
q = int(input())
a = []
for _ in range(q):
    query = list(map(int, input().split()))
    if query[0] == 1:
        a.append(query[1])
    else:
        k = query[1]
        print(a[-k])