import sys
data = sys.stdin.read().split()
N = int(data[0])
K = int(data[1])
S = data[2]

ans = 0
i = 0
while i < N:
    if S[i] == 'O':
        start = i
        while i < N and S[i] == 'O':
            i += 1
        len_run = i - start
        ans += len_run // K
    else:
        i += 1

print(ans)