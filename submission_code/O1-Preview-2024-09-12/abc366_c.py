# YOUR CODE HERE
import sys
from collections import defaultdict

input = sys.stdin.readline

Q = int(sys.stdin.readline())
count = defaultdict(int)
res = []
distinct = 0

for _ in range(Q):
    query = sys.stdin.readline().split()
    if query[0] == '1':
        x = int(query[1])
        if count[x] == 0:
            distinct +=1
        count[x] +=1
    elif query[0] == '2':
        x = int(query[1])
        count[x] -=1
        if count[x] == 0:
            distinct -=1
    else:
        res.append(str(distinct))

print('
'.join(res))