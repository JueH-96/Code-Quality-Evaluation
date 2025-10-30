N, S = map(int, input().split())
A = list(map(int, input().split()))

cumsum, cumsum_mod = [0], [0]*S
for i, a in enumerate(A):
    cumsum.append(cumsum[-1] + a)
    cumsum_mod[(cumsum[-1] % S)] = i+1

for i, c in enumerate(cumsum):
    if c % S in cumsum_mod and cumsum_mod[c % S] < i:
        print('Yes')
        break
else:
    print('No')