n, m = map(int, input().split())
c = input().split()
d = input().split()
p = list(map(int, input().split()))

price_map = {}
for i in range(m):
    color = d[i]
    price_map[color] = p[i+1]
p0 = p[0]

total = 0
for color in c:
    total += price_map.get(color, p0)

print(total)