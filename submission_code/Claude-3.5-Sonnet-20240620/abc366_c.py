# YOUR CODE HERE
from collections import Counter

bag = Counter()
Q = int(input())

for _ in range(Q):
    query = input().split()
    
    if query[0] == '1':
        bag[int(query[1])] += 1
    elif query[0] == '2':
        bag[int(query[1])] -= 1
        if bag[int(query[1])] == 0:
            del bag[int(query[1])]
    else:  # query[0] == '3'
        print(len(bag))