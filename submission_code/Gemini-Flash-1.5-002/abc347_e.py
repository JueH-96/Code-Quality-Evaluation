# YOUR CODE HERE
n, q = map(int, input().split())
x = list(map(int, input().split()))

a = [0] * n
s = set()

for val in x:
    if val in s:
        s.remove(val)
    else:
        s.add(val)
    
    for i in range(n):
        if i + 1 in s:
            a[i] += len(s)

print(*a)