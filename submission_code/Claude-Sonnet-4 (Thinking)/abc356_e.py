from collections import defaultdict

n = int(input())
a = list(map(int, input().split()))

total = 0
count = defaultdict(int)

for i in range(n - 1, -1, -1):
    for v in count:
        if v >= a[i]:
            total += count[v] * (v // a[i])
        else:
            total += count[v] * (a[i] // v)
    
    count[a[i]] += 1

print(total)