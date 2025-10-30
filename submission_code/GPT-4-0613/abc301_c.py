from collections import Counter

S = input().strip()
T = input().strip()

S = S.replace('@', 'atcoder')
T = T.replace('@', 'atcoder')

if Counter(S) == Counter(T):
    print('Yes')
else:
    print('No')