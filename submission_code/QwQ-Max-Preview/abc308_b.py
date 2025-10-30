n, m = map(int, input().split())
c_list = input().split()
d_list = input().split()
p = list(map(int, input().split()))

p0 = p[0]
price_dict = {}
for i in range(m):
    price_dict[d_list[i]] = p[i + 1]

total = 0
for color in c_list:
    total += price_dict.get(color, p0)

print(total)