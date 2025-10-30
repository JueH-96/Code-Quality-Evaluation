# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))
mod = 998244353

def f(x, y):
    return int(str(x) + str(y))

total = 0
for i in range(n):
    for j in range(i + 1, n):
        total += f(a[i], a[j])
        total %= mod

print(total)