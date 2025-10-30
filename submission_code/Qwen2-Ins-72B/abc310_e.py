import sys
input = sys.stdin.readline

N = int(input())
S = input().strip()

ans = 0
prev = 0
for i in range(N):
    if S[i] == '1':
        ans += prev * (N - i) + (N - i)
        prev = 0
    else:
        ans += prev * (N - i)
        prev += 1
print(ans)