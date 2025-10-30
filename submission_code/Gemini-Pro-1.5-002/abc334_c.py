# YOUR CODE HERE
def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))
    
    counts = [2] * n
    for x in a:
        counts[x-1] -= 1
    
    remaining_socks = []
    for i in range(n):
        for _ in range(counts[i]):
            remaining_socks.append(i+1)
            
    num_pairs = (2 * n - k) // 2
    
    if num_pairs == 0:
        print(0)
        return

    dp = [[float('inf')] * (num_pairs + 1) for _ in range(len(remaining_socks) + 1)]
    dp[0][0] = 0
    
    for i in range(1, len(remaining_socks) + 1):
        for j in range(num_pairs + 1):
            if j == 0:
                dp[i][j] = 0
            else:
                dp[i][j] = dp[i-1][j]
                if i >= 2:
                    dp[i][j] = min(dp[i][j], dp[i-2][j-1] + abs(remaining_socks[i-1] - remaining_socks[i-2]))
    
    print(dp[len(remaining_socks)][num_pairs])

solve()