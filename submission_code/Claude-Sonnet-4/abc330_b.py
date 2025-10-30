# YOUR CODE HERE
n, l, r = map(int, input().split())
a = list(map(int, input().split()))

result = []
for ai in a:
    if ai < l:
        xi = l
    elif ai > r:
        xi = r
    else:
        xi = ai
    result.append(xi)

print(' '.join(map(str, result)))