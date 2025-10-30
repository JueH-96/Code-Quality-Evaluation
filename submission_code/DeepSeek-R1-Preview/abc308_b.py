n, m = map(int, input().split())
c_list = input().split()
d_list = input().split()
p_list = list(map(int, input().split()))

color_price = {}
for i in range(m):
    color = d_list[i]
    price = p_list[i + 1]
    color_price[color] = price

total = 0
for c in c_list:
    if c in color_price:
        total += color_price[c]
    else:
        total += p_list[0]

print(total)