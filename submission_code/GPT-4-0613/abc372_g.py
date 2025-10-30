# YOUR CODE HERE
import sys
from math import gcd
from heapq import *
from collections import deque
input = sys.stdin.readline
t = int(input())
for _ in range(t):
    n = int(input())
    abc = [list(map(int,input().split())) for _ in range(n)]
    abc.sort(key = lambda x: -x[2])
    l = abc[0][2]
    ans = 0
    q = deque()
    for a,b,c in abc:
        if c < l:
            while q and q[0][0] < c:
                _,x,y = heappop(q)
                l = min(l,x+y)
        ans += l - a - b
        while q and q[0][0] <= a:
            _,x,y = heappop(q)
            l = min(l,x+y)
        g = gcd(a,b)
        heappush(q,(a//g*g if a%g*2>=g else (a//g-1)*g,a,b))
    print(ans)