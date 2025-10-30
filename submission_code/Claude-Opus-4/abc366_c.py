# YOUR CODE HERE
from collections import defaultdict

Q = int(input())
bag = defaultdict(int)

for _ in range(Q):
    query = list(map(int, input().split()))
    
    if query[0] == 1:
        # Add a ball with integer x
        x = query[1]
        bag[x] += 1
    
    elif query[0] == 2:
        # Remove a ball with integer x
        x = query[1]
        bag[x] -= 1
        if bag[x] == 0:
            del bag[x]
    
    elif query[0] == 3:
        # Print number of different integers
        print(len(bag))