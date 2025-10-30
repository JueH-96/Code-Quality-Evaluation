N, K = map(int, input().split())
A = list(map(int, input().split()))

rides = 0
seats = K
for i in range(N):
    if seats >= A[i]:
        seats -= A[i]
    else:
        rides += 1
        seats = K - A[i]
rides += 1
print(rides)