n, m = map(int, input().split())
c_list = input().split()
d_list = input().split()
p_list = list(map(int, input().split()))

price_dict = {d: p for d, p in zip(d_list, p_list[1:])}

total = 0
for c in c_list:
    if c in price_dict:
        total += price_dict[c]
    else:
        total += p_list[0]

print(total)