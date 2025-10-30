import math
from collections import defaultdict

def is_square(n):
    if n == 0:
        return True
    x = int(math.sqrt(n))
    return x*x == n

N = int(input().strip())
A = list(map(int, input().strip().split()))

count = defaultdict(int)
for a in A:
    count[a] += 1

ans = 0
for a in count:
    if a == 0:
        ans += count[a] * (count[a] - 1) // 2
    else:
        for b in range(a+1, 2*10**5+1):
            if is_square(a*b):
                ans += count[a] * count[b]
print(ans)