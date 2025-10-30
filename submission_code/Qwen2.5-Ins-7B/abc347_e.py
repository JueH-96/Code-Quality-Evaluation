# YOUR CODE HERE
n, q = map(int, input().split())
a = [0] * n
s = set()

for x in map(int, input().split()):
    if x in s:
        s.remove(x)
    else:
        s.add(x)
    a = [a[i] + len(s) for i in range(n)]

print(*a)