N = int(input())
M = N//2
nums = [str(i)+"
"
        if j%2==i%2 and i!=M and j!=M and i+j!=N-1 and i-j!=N-1
        else str(i*N+j+2)+"
"
        if i==M or j==M or i+j==N-1 or i==j+N-1
        else str(i*N+j+1)+"
"
        for i in range(N) for j in range(N)]
nums[N*M+N//2+M]= "T
"
for i in range(M):
    nums[N*(M-i-1)+M-i-1], nums[N*M+N-i], nums[N*M+i], nums[N*(M+i)+N-i-1] =nums[N*M+N-i],nums[N*(M+i)+N-i-1] ,nums[N*(M-i-1)+M-i-1], nums[N*M+i]
nums[N*M+N//2+M-1], nums[N*M+N//2+M+1], nums[N*M+N//2+M+N], nums[N*M+N//2+M-N] =nums[N*M+N//2+M+N],nums[N*M+N//2+M-N] ,nums[N*M+N//2+M-1], nums[N*M+N//2+M+1]
print("
".join("".join(i for i in nums[j:j+N]) for j in range(0, N*N, N)))