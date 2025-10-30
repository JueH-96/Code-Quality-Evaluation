import re

N, M = map(int, input().split())
S = input()
T = input()

pattern = '.' * (N - M) + T
pattern = pattern.replace(T, 'T{1,' + str(N - M) + '}')
pattern = pattern.replace('T', T)

if re.fullmatch(pattern, S):
    print('Yes')
else:
    print('No')