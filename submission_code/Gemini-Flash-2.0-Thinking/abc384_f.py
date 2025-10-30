def f(x):
    while x % 2 == 0:
        x //= 2
    return x

n = int(input())
a = list(map(int, input().split()))

total_sum = 0
for i in range(n):
    for j in range(i, n):
        total_sum += f(a[i] + a[j])

print(total_sum)