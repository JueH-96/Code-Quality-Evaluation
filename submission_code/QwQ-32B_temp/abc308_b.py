n, m = map(int, input().split())
c_list = input().split()
d_list = input().split()
p_values = list(map(int, input().split()))
p0 = p_values[0]
price_dict = {d: p_values[i+1] for i, d in enumerate(d_list)}
total = 0
for c in c_list:
    if c in price_dict:
        total += price_dict[c]
    else:
        total += p0
print(total)