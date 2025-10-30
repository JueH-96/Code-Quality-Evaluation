N, M, P = map(int, input().split())

ans = 0
while M <= N:
    ans += 1
    M += P

print(ans)