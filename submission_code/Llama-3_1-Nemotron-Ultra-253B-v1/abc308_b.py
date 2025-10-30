n, m = map(int, input().split())
c = input().split()
d = input().split()
p = list(map(int, input().split()))

color_price = {d[i]: p[i+1] for i in range(m)}
total = 0

for color in c:
    total += color_price.get(color, p[0])

print(total)