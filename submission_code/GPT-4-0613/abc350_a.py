S = input().strip()
if S == 'ABC316':
    print('No')
elif int(S[3:]) <= 349:
    print('Yes')
else:
    print('No')