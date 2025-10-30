import sys

Q = int(input())
bag = set()
output = []

for _ in range(Q):
    query = list(map(int, input().split()))
    if query[0] == 1:
        bag.add(query[1])
    elif query[0] == 2:
        bag.remove(query[1])
    else:
        output.append(len(bag))

for o in output:
    print(o)