N = int(input())
A = list(map(int, input().split()))

ans = 0
for i in range(N-1):
    for j in range(i+1, N):
        xor = 0
        for k in range(i, j+1):
            xor ^= A[k]
        ans += xor

print(ans)