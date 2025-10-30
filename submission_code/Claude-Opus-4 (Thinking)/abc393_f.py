# YOUR CODE HERE
import bisect

n, q = map(int, input().split())
a = list(map(int, input().split()))

for _ in range(q):
    r, x = map(int, input().split())
    
    tails = []
    for i in range(r):
        num = a[i]
        if num > x:
            continue
        
        pos = bisect.bisect_left(tails, num)
        
        if pos == len(tails):
            tails.append(num)
        else:
            tails[pos] = num
    
    print(len(tails))