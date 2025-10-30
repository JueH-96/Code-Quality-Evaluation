N = int(input())
AB = [list(map(int, input().split())) for _ in range(N)]

# B - A の降順でソート
AB.sort(key=lambda x: x[1] - x[0], reverse=True)

# A の累積和
A = [0]
for a, _ in AB:
    A.append(A[-1] + a)

ans = 0
for i in range(N):
    ans = max(ans, A[i] + AB[i][1])
print(ans)