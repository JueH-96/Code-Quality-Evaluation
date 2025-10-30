# YOUR CODE HERE
n = int(input())
p = list(map(int, input().split()))
a = list(map(int, input().split()))

p = [x - 1 for x in p]
a = [x - 1 for x in a]

applied_p = [a]
current_a = a[:]
for _ in range(n):
    next_a = [0] * n
    for i in range(n):
        next_a[i] = current_a[p[i]]
    current_a = next_a
    applied_p.append(current_a)

min_a = applied_p[0]
for ap in applied_p:
    if ap < min_a:
        min_a = ap

print(*[x + 1 for x in min_a])