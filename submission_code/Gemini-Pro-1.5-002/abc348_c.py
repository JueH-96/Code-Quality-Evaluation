# YOUR CODE HERE
n = int(input())
beans = []
for _ in range(n):
    a, c = map(int, input().split())
    beans.append((a, c))

colors = {}
for a, c in beans:
    if c not in colors:
        colors[c] = []
    colors[c].append(a)

max_min_deliciousness = 0
for c in colors:
    min_deliciousness = min(colors[c])
    max_min_deliciousness = max(max_min_deliciousness, min_deliciousness)

print(max_min_deliciousness)