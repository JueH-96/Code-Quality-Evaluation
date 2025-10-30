# YOUR CODE HERE
n, m, d = map(int, input().split())
a = list(map(int, input().split()))
b = list(map(int, input().split()))

max_sum = -1
for x in a:
    for y in b:
        if abs(x - y) <= d:
            max_sum = max(max_sum, x + y)

print(max_sum)