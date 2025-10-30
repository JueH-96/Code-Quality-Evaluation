N = int(input())
A = list(map(int, input().split()))

ans = 0
for i in range(N-1):
    for j in range(i+1, N):
        max_val = max(A[i], A[j])
        min_val = min(A[i], A[j])
        ans += max_val // min_val

print(ans)