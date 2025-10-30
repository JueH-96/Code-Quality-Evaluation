import math
N = int(input())
ans = 0
while N >= 2:
    ans += N
    N = math.floor(N/2) + math.ceil(N/2)
print(ans)