import sys
from collections import deque

H, W, X, Y = map(int, input().split())
# adjust the tile
Y-=1
X-=1
# since lines beginning with '#' end with '#'
# we just need to store the middle portion containing ' .@'
# and we can iterate by indexing them by Python's list slicing
temp = []
for _ in range(H):
    temp.append(list(input()[1:W-1]))
# the same adjustement can be applied to T since 'T' is 0-indexed
T = input()
S = temp

dxdy_chars = [(0,1,'R'),(0,-1,'L'),(1,0,'D'),(-1,0,'U')]
T = deque([*T])

st = (X, Y, 0)
visited = set()
while T:
    ch = T.popleft()
    for dx,dy,uchar in dxdy_chars:
        if ch == uchar and 0 <= st[0] + dx < len(S) and 0 <= st[1]+dy < len(S[0]) and S[st[0]+dx][st[1]+dy] != '#':
            st = (st[0] + dx, st[1] + dy, st[2])
            visited.add(st)
            break
print(st[0]+1, st[1]+1, len(visited))