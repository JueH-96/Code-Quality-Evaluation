# YOUR CODE HERE
n = int(input())
q = list(map(int, input().split()))
a = list(map(int, input().split()))
b = list(map(int, input().split()))

min_servings = float('inf')
for i in range(n):
    min_servings = min(min_servings, q[i] // a[i])

max_servings = 0
for x in range(min_servings + 1):
    y = min_servings - x
    possible = True
    for i in range(n):
        if x * a[i] + y * b[i] > q[i]:
            possible = False
            break
    if possible:
        max_servings = max(max_servings, x + y)

print(max_servings)