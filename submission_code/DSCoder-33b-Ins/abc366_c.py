# YOUR CODE HERE
from collections import Counter

Q = int(input())
bag = Counter()

for _ in range(Q):
    query = list(map(int, input().split()))
    if query[0] == 1:
        bag[query[1]] += 1
    elif query[0] == 2:
        bag[query[1]] -= 1
        if bag[query[1]] == 0:
            del bag[query[1]]
    else:
        print(len(bag))