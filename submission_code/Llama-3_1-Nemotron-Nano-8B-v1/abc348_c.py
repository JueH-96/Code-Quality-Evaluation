# YOUR CODE HERE
n = int(input())
color_min = {}

for _ in range(n):
    a, c = map(int, input().split())
    if c not in color_min or a < color_min[c]:
        color_min[c] = a

print(max(color_min.values()))