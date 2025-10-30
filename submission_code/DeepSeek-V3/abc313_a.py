# YOUR CODE HERE
n = int(input())
p = list(map(int, input().split()))
p1 = p[0]
others = p[1:]
max_other = max(others)
if p1 > max_other:
    print(0)
else:
    print(max_other - p1 + 1)