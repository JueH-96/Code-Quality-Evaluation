N = int(input().strip())
S = input().strip()

cnt = [0]*(N+1)
for i in range(N):
    cnt[i+1] = cnt[i] + (S[i] == 'R')

ans = min(cnt[i] + (N-i) - (cnt[N] - cnt[i]) for i in range(N+1))
print(N - ans)