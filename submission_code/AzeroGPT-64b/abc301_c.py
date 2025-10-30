S = input().replace('@', 'a')
T = input().replace('@', 'a')

c = Counter(S)
for c_ in Counter(T).items():
    if c_[0] not in c or c_[1] > c[c_[0]]:
        print('No')
        exit(0)
print('Yes')