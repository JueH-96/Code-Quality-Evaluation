import bisect

N = int(input())
A = list(map(int,input().split()))
Q = int(input())
LR = [list(map(int,input().split())) for _ in range(Q)]

B = [0]*(N+1)
for i in range(1,N+1,2):
    B[i+1] = B[i-1] + A[i] - A[i-1]

for l,r in LR:
    lb = bisect.bisect_right(A,l)
    rb = bisect.bisect_left(A,r)
    if lb % 2 == 0:
        ans = B[rb]
        if rb % 2 == 1:
            ans += r - A[rb-1]
    else:
        ans = B[rb] - (A[lb-1] - l)
        if rb % 2 == 1:
            ans += r - A[rb-1]
    print(ans)