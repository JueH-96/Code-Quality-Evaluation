import math

n = int(input())
m = math.ceil(math.log2(n))
print(m)
for i in range(m):
    bottles = []
    for j in range(1, n + 1):
        if (j >> i) & 1:
            bottles.append(j)
    print(len(bottles), *bottles)
s = input()
ans = 0
for i in range(m):
    if s[i] == '1':
        ans |= (1 << i)
if ans == 0:
    print(n)
else:
    print(ans)