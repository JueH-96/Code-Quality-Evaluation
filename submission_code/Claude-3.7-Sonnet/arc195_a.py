def subsequence_count(A, B, N, M):
    dp = [0 for _ in range(M+1)]
    
    # Base case: there's 1 way to form an empty subsequence
    dp[0] = 1
    
    for i in range(1, N+1):
        for j in range(M, 0, -1):
            if A[i-1] == B[j-1]:
                dp[j] += dp[j-1]
    
    return dp[M]

N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

count = subsequence_count(A, B, N, M)
if count >= 2:
    print("Yes")
else:
    print("No")