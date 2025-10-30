from collections import defaultdict

Q = int(input())
queries = [input().split() for _ in range(Q)]

bag = defaultdict(int)
unique = set()

for q in queries:
    if q[0] == '1':
        x = int(q[1])
        bag[x] += 1
        if x not in unique:
            unique.add(x)
    elif q[0] == '2':
        x = int(q[1])
        bag[x] -= 1
        if bag[x] == 0:
            unique.remove(x)
    else:
        print(len(unique))