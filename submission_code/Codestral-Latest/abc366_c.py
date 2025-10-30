import sys
from collections import defaultdict

input = sys.stdin.read
data = input().split()

Q = int(data[0])
index = 1
bag = defaultdict(int)
result = []

for _ in range(Q):
    query_type = int(data[index])
    if query_type == 1:
        x = int(data[index + 1])
        bag[x] += 1
        index += 2
    elif query_type == 2:
        x = int(data[index + 1])
        if bag[x] == 1:
            del bag[x]
        else:
            bag[x] -= 1
        index += 2
    else:
        result.append(len(bag))
        index += 1

print("
".join(map(str, result)))