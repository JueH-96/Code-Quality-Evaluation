# YOUR CODE HERE
def main():
    T = input().strip()
    N = int(input())
    bags = []
    for _ in range(N):
        parts = input().split()
        A_i = int(parts[0])
        S_i = parts[1:]
        bags.append(S_i)
    
    len_T = len(T)
    # dp[i][j] represents the minimum cost to match the first j characters of T using the first i bags
    # Initialize with infinity
    INF = float('inf')
    dp = [[INF] * (len_T + 1) for _ in range(N + 1)]
    dp[0][0] = 0
    
    for i in range(1, N + 1):
        for j in range(len_T + 1):
            # Option 1: Do nothing, carry over the previous cost
            if dp[i-1][j] < INF:
                dp[i][j] = min(dp[i][j], dp[i-1][j])
            # Option 2: Choose a string from the current bag
            for s in bags[i-1]:
                len_s = len(s)
                if j >= len_s and T[j-len_s:j] == s:
                    if dp[i-1][j-len_s] < INF:
                        dp[i][j] = min(dp[i][j], dp[i-1][j-len_s] + 1)
    
    if dp[N][len_T] < INF:
        print(dp[N][len_T])
    else:
        print(-1)

if __name__ == "__main__":
    main()