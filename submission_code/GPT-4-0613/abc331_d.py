# YOUR CODE HERE
import sys
input = sys.stdin.readline

N, Q = map(int, input().split())
P = [list(input().strip()) for _ in range(N)]
cnt = [[0]*(N*2+1) for _ in range(N*2+1)]
for i in range(N*2):
    for j in range(N*2):
        cnt[i+1][j+1] = cnt[i+1][j] + cnt[i][j+1] - cnt[i][j] + (P[i%N][j%N] == 'B')
for _ in range(Q):
    a, b, c, d = map(int, input().split())
    a %= (N*2)
    b %= (N*2)
    c %= (N*2)
    d %= (N*2)
    if a > c:
        a, c = c, a
    if b > d:
        b, d = d, b
    ans = 0
    ans += (cnt[c][d] - cnt[a][d] - cnt[c][b] + cnt[a][b]) * ((c-a)//(N*2)) * ((d-b)//(N*2))
    a %= (N*2)
    b %= (N*2)
    c %= (N*2)
    d %= (N*2)
    if a > c:
        a, c = c, a
    if b > d:
        b, d = d, b
    ans += (cnt[c][d] - cnt[a][d] - cnt[c][b] + cnt[a][b])
    print(ans)