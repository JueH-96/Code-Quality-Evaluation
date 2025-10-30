n = int(input())
a = list(map(int, input().split()))
b = list(map(int, input().split()))

max_a = a[0]
for i in range(1, n):
    if a[i] > max_a:
        max_a = a[i]

max_b = b[0]
for j in range(1, n):
    if b[j] > max_b:
        max_b = b[j]

print(max_a + max_b)