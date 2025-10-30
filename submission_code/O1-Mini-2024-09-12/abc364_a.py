N = int(input())
S = [input().strip() for _ in range(N)]
prev = ''

for i in range(N):
    current = S[i]
    if current == 'sweet' and prev == 'sweet':
        if i == N - 1:
            print('Yes')
        else:
            print('No')
        exit()
    prev = current
print('Yes')