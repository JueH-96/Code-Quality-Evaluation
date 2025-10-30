from collections import Counter

N = int(input().strip())
A = list(map(int, input().strip().split()))

A.sort(reverse=True)
A.append(-1)

counter = Counter()
res = 0

for a in A:
    if counter[a] > 0:
        counter[a] -= 1
    else:
        res += 1
        counter[a - 1] += 1

print(res)