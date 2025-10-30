# YOUR CODE HERE
import math

n = int(input())
m = math.ceil(math.log2(n))
print(m)
friends = []
for i in range(m):
    group = []
    for j in range(n):
        if (j >> i) & 1:
            group.append(j + 1)
    print(len(group), *group)
    friends.append(group)

s = input()
ans = 0
for i in range(m):
    if s[i] == '1':
        ans += 2**i
print(ans)