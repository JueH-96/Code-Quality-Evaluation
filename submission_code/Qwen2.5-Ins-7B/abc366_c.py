# YOUR CODE HERE
from collections import Counter

Q = int(input())
queries = []
for _ in range(Q):
    queries.append(list(map(int, input().split())))

bag = Counter()
for query in queries:
    if query[0] == 1:
        bag[query[1]] += 1
    elif query[0] == 2:
        bag[query[1]] -= 1
        if bag[query[1]] == 0:
            del bag[query[1]]
    else:
        print(len(bag))