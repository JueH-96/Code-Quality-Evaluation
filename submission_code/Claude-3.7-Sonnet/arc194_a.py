def solve():
    N = int(input())
    A = list(map(int, input().split()))
    
    # dp[i][j] = maximum sum after processing the first i elements, where the size of S is j
    dp = {}
    dp[(0, 0)] = 0
    
    # seq[i][j] = sequence corresponding to dp[i][j]
    seq = {}
    seq[(0, 0)] = []
    
    for i in range(1, N + 1):
        for j in range(0, i + 1):
            # Try to append A[i-1]
            if j > 0 and (i-1, j-1) in dp:
                append_sum = dp[(i-1, j-1)] + A[i-1]
                if (i, j) not in dp or append_sum > dp[(i, j)]:
                    dp[(i, j)] = append_sum
                    seq[(i, j)] = seq[(i-1, j-1)].copy()
                    seq[(i, j)].append(A[i-1])
            
            # Try to delete the last element of S
            if j < i and (i-1, j+1) in dp and seq[(i-1, j+1)]:
                delete_sum = dp[(i-1, j+1)] - seq[(i-1, j+1)][-1]
                if (i, j) not in dp or delete_sum > dp[(i, j)]:
                    dp[(i, j)] = delete_sum
                    seq[(i, j)] = seq[(i-1, j+1)].copy()
                    seq[(i, j)].pop()
    
    max_sum_val = -float('inf')
    for j in range(N + 1):
        if (N, j) in dp:
            max_sum_val = max(max_sum_val, dp[(N, j)])
    
    print(max_sum_val)

solve()