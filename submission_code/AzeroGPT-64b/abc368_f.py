from math import sqrt

n = int(input())
a = list(map(int, input().split()))

def nimber(n):
    if n == 1:
        return 0
    x = int(sqrt(n))
    if x * x == n:
        x -= 1
    return n - x * (x + 1)

nim = 0
for aa in a:
    nim ^= nimber(aa - 1)

print('Anna' if nim else 'Bruno')