import sys
from collections import defaultdict

N = int(input())
A = list(map(int, input().split()))
Q = int(input())

prev = defaultdict(int)
next_ = defaultdict(int)

for i in range(N-1):
    prev[A[i+1]] = A[i]
    next_[A[i]] = A[i+1]

l = [A[0]]
cur = A[0]

for _ in range(Q):
    query = list(map(int, input().split()))
    
    if query[0] == 1:
        x, y = query[1], query[2]
        
        if y in next_:
            next_y = next_[y]
            prev[next_y] = x
            next_[y] = x
            next_[x] = y
            if cur == x:
                cur = y
        else:
            next_[x] = y
            next_[cur] = y
            prev[y] = cur
            cur = y
        
        del prev[y]
    else:
        x = query[1]
        
        if x in prev:
            prev_of_x = prev[x]
            if x == cur:
                cur = prev_of_x
            del prev[prev_of_x]
        else:
            prev_of_x = None
        
        if x in next_:
            next_of_x = next_[x]
            del prev[next_of_x]
            if x == cur:
                cur = next_of_x
        elif x == cur:
            cur = prev_of_x
        
        del next_[x]

    while cur in next_:
        cur = next_[cur]
        l.append(cur)
    
prev = defaultdict(int)
for a in l[:N-1]:
    next_[a] = l[l.index(a)+1]
for a in l[:N]:
    prev[a] = l[l.index(a)-1] if l.index(a) > 0 else None

for _ in range(Q):
    query = list(map(int, input().split()))
    
    if query[0] == 1:
        x, y = query[1], query[2]
        
        if y in prev:
            prev_y = prev[y]
            next_[prev_y] = x
            if y == cur:
                cur = x
            prev[y] = x
            prev[x] = y
            if cur == x:
                cur = y
        else:
            prev[y] = x
            prev[x] = y
            next_[x] = y
            if cur == x:
                cur = y
        
    else:
        x = query[1]
        
        if x in next_:
            next_of_x = next_[x]
            del prev[next_of_x]
            if x == cur:
                cur = next_of_x
        elif x == cur:
            cur = prev[x]
        
        del next_[x]
        if x in prev:
            del prev[x]

while cur in next_:
    cur = next_[cur]
    l.append(cur)

print(' '.join(map(str, l)))