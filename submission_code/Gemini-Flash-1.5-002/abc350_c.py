# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

ops = []
for i in range(n):
    if a[i] != i + 1:
        j = a.index(i + 1)
        ops.append((i + 1, j + 1))
        a[i], a[j] = a[j], a[i]

print(len(ops))
for op in ops:
    print(op[0], op[1])