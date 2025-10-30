# YOUR CODE HERE
s1, s2 = input().split()
t1, t2 = input().split()

def dist(p1, p2):
    p = 'ABCDE'
    d = abs(p.index(p1) - p.index(p2))
    return min(d, 5 - d)

if dist(s1, s2) == dist(t1, t2):
    print('Yes')
else:
    print('No')