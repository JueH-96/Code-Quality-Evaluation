# YOUR CODE HERE
n = int(input())
p = list(map(int, input().split()))

p1 = p[0]
max_others = max(p[1:])

if p1 > max_others:
    print(0)
else:
    print(max_others - p1 + 1)