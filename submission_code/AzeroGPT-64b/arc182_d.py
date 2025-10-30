from collections import deque

N,M = map(int,input().split())
A = list(map(int,input().split()))
B = list(map(int,input().split()))
A1,B1 = A[0],B[0]
dp = [[0,M,M] for _ in range(N)]
isOK = [True]*M
ans = 0
q = deque([A1])
isOK[A1] = False
if B1==A1:
    dp[0][0] = 0
while q:
    bef = q.popleft()
    for i in range(2):
        ind = (bef+i-1)%M
        if not isOK[ind]:
            continue
        isOK[ind] = False
        q.append(ind)
        cur = 0
        ans_now = 0
        for j in range(N):
            A_now = A[j]
            B_now = B[j]
            if ind==B_now:
                dp[j][0] = 1
                if A_now!=ind:
                    if dp[j-1][2]==M:
                        ans_now = M+1
                    elif dp[j-1][2]==1:
                        ans_now = M-1
                    else:
                        ans_now = M
                    cur += ans_now
                    #print(f'a1{j,ans_now}')
                else:
                    if i!=1 :
                        dp[j][1] = 1
                        cur += dp[j-1][2]
                    if dp[j-1][1]==M and i!=1:
                        ans_now = M+1
                    elif dp[j-1][1]==1 and i!=1:
                        ans_now = M-1
                    elif M-1 not in dp[j-1] and dp[j-1][0]==1:
                        ans_now = max(M-1,1)
                    else:
                        ans_now = 1
                    cur += ans_now
                    #print(f'a2{j,ans_now}')
            else:
                if dp[j-1][0]==1:
                    dp[j][0] = 2
                    cur += (M if A_now!=ind else 1)
                    #print(f'b1{j,M}')
                if dp[j-1][0]!=M:
                    dp[j][1] = 2
                    cur += (M+1 if A_now!=ind else M)
                    #print(f'b2{j,M+1}')
            if j==N-1:
                ans = min(ans,cur)
print(ans)