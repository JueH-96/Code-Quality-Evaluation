# YOUR CODE HERE
n = int(input())
color_min = {}
for _ in range(n):
    a, c = map(int, input().split())
    if c not in color_min:
        color_min[c] = a
    else:
        color_min[c] = min(color_min[c], a)

result = max(color_min.values())
print(result)