N, M = map(int, input().split())
A_l, B_l = zip(*(map(str.split, (input() for _ in range(M)))))

taro = ['No'] * M
s = set()
f = {a:b for a, b in zip(A_l, B_l)}

for i in range(M):
    if f[A_l[i]] == 'M':
        if A_l[i] not in s:
            s.add(A_l[i])
            taro[i] = 'Yes'
        f[A_l[i]] = 'F'

print('
'.join(taro))