# YOUR CODE HERE
N, M = map(int, input().strip().split())
C = input().strip().split()
D = input().strip().split()
P_all = list(map(int, input().strip().split()))
P_0 = P_all[0]
P_i = P_all[1:]

price_dict = {}
for d_i, p_i in zip(D, P_i):
    price_dict[d_i] = p_i

total = 0
for c_i in C:
    total += price_dict.get(c_i, P_0)

print(total)