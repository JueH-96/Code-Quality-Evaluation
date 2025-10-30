# YOUR CODE HERE
n, m = map(int, input().split())
s = [input() for _ in range(n)]

ans = float('inf')
for i in range(1 << n):
    stands = []
    for j in range(n):
        if (i >> j) & 1:
            stands.append(j)
    
    flavors = set()
    for stand_index in stands:
        for k in range(m):
            if s[stand_index][k] == 'o':
                flavors.add(k)
    
    if len(flavors) == m:
        ans = min(ans, len(stands))

print(ans)