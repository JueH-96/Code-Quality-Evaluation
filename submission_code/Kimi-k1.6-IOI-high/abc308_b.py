n, m = map(int, input().split())
c = input().split()
d = input().split()
p = list(map(int, input().split()))

p0 = p[0]
price_map = {d[i]: p[i+1] for i in range(m)}

total = 0
for color in c:
    total += price_map.get(color, p0)

print(total)