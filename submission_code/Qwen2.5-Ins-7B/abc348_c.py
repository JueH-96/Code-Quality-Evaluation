# YOUR CODE HERE
n = int(input())
beans = {}
for _ in range(n):
    a, c = map(int, input().split())
    if c in beans:
        beans[c].append(a)
    else:
        beans[c] = [a]
max_min_deliciousness = 0
for colors in beans.values():
    colors.sort()
    for i in range(len(colors) - 1):
        max_min_deliciousness = max(max_min_deliciousness, colors[i])
    max_min_deliciousness = max(max_min_deliciousness, colors[-1])
print(max_min_deliciousness)