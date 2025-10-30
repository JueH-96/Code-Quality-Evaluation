from collections import defaultdict

Q = int(input())
bag = defaultdict(int)

for _ in range(Q):
    query = input().split()
    query_type = int(query[0])

    if query_type == 1:
        x = int(query[1])
        bag[x] += 1
    elif query_type == 2:
        x = int(query[1])
        bag[x] -= 1
        if bag[x] == 0:
            del bag[x]
    elif query_type == 3:
        print(len(bag))