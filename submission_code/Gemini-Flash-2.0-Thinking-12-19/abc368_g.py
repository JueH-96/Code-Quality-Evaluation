def solve():
    n = int(input())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    q = int(input())
    
    for _ in range(q):
        query = list(map(int, input().split()))
        query_type = query[0]
        if query_type == 1:
            i, x = query[1], query[2]
            a[i-1] = x
        elif query_type == 2:
            i, x = query[1], query[2]
            b[i-1] = x
        elif query_type == 3:
            l, r = query[1], query[2]
            length = r - l + 1
            dp = [[0] * (length + 1) for _ in range(length + 1)]
            
            dp[0][0] = 0
            
            for i in range(1, length + 1):
                current_a = a[l+i-2]
                current_b = b[l+i-2]
                dp[i][0] = dp[i-1][0] + current_a
                for j in range(1, i + 1):
                    dp[i][j] = max(dp[i-1][j] + current_a, dp[i-1][j-1] * current_b)
                    
            max_value = 0
            for j in range(length + 1):
                max_value = max(max_value, dp[length][j])
            print(max_value)

if __name__ == '__main__':
    solve()