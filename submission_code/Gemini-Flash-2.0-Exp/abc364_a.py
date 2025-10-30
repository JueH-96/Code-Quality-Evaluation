N = int(input())
S = [input() for _ in range(N)]

sick = False
for i in range(N - 1):
    if S[i] == 'sweet' and S[i+1] == 'sweet':
        sick = True
        break

if sick:
    print('No')
else:
    print('Yes')