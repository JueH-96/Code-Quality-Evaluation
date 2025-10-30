# YOUR CODE HERE
def distance(p1, p2):
    points = 'ABCDE'
    n = len(points)
    i1, i2 = points.index(p1), points.index(p2)
    d = min((i2 - i1) % n, (i1 - i2) % n)
    return min(d, n - d)

s1, s2 = input().strip()
t1, t2 = input().strip()

if distance(s1, s2) == distance(t1, t2):
    print("Yes")
else:
    print("No")