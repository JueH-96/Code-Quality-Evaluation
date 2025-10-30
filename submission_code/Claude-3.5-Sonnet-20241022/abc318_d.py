def read_input():
    N = int(input())
    D = [[0] * N for _ in range(N)]
    for i in range(N-1):
        row = list(map(int, input().split()))
        for j in range(len(row)):
            D[i][i+j+1] = row[j]
            D[i+j+1][i] = row[j]
    return N, D

def solve(N, D):
    # dp[S] represents max weight when vertices in set S are used
    dp = {}
    
    def rec(used):
        if used in dp:
            return dp[used]
            
        if bin(used).count('1') >= N-1:
            return 0
            
        res = 0
        # Try all pairs of unused vertices
        for i in range(N):
            if used & (1 << i):
                continue
            for j in range(i+1, N):
                if used & (1 << j):
                    continue
                    
                # Use edge between i and j
                next_used = used | (1 << i) | (1 << j)
                res = max(res, D[i][j] + rec(next_used))
                
        dp[used] = res
        return res
        
    return rec(0)

N, D = read_input()
print(solve(N, D))