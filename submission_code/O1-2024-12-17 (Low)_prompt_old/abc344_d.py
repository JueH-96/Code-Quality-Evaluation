def solve():
    import sys
    
    input_data = sys.stdin.read().strip().split()
    T = input_data[0]
    N = int(input_data[1])
    
    idx = 2
    bags = []
    for _ in range(N):
        A_i = int(input_data[idx])
        idx += 1
        strings = input_data[idx:idx+A_i]
        idx += A_i
        bags.append(strings)
    
    # Length of the target string T
    lenT = len(T)
    
    # dp[i][l] = minimum cost to achieve prefix of length l of T after processing i bags
    INF = float('inf')
    dp = [[INF]*(lenT+1) for _ in range(N+1)]
    dp[0][0] = 0  # No cost to form empty string with 0 bags
    
    for i in range(N):
        for l in range(lenT+1):
            if dp[i][l] == INF:
                continue
            
            # Option 1: Do nothing
            dp[i+1][l] = min(dp[i+1][l], dp[i][l])
            
            # Option 2: Pay 1 yen and pick one string from bag i
            for s in bags[i]:
                new_len = l + len(s)
                if new_len <= lenT and T[l:new_len] == s:
                    dp[i+1][new_len] = min(dp[i+1][new_len], dp[i][l] + 1)
    
    ans = dp[N][lenT]
    print(ans if ans != INF else -1)

def main():
    solve()

if __name__ == "__main__":
    main()