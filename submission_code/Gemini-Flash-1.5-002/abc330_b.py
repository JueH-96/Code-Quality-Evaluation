# YOUR CODE HERE
n, l, r = map(int, input().split())
a = list(map(int, input().split()))

results = []
for x in a:
    if x < l:
        results.append(l)
    elif x > r:
        results.append(r)
    else:
        results.append(x)

print(*results)