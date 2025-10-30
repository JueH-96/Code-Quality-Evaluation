N, D = map(int, input().split())
S = list(input())

for day in range(D):
    # Find the rightmost cookie
    for i in range(N - 1, -1, -1):
        if S[i] == '@':
            S[i] = '.'
            break

print(''.join(S))