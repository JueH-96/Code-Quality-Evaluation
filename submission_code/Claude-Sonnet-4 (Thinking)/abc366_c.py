from collections import defaultdict

Q = int(input())
bag = defaultdict(int)

for _ in range(Q):
    query = input().split()
    
    if query[0] == '1':
        x = int(query[1])
        bag[x] += 1
    elif query[0] == '2':
        x = int(query[1])
        bag[x] -= 1
        if bag[x] == 0:
            del bag[x]
    else:  # query[0] == '3'
        print(len(bag))