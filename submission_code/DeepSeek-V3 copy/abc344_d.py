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
    # dp[i] represents the minimum cost to construct the first i characters of T
    INF = float('inf')
    dp = [INF] * (len_T + 1)
    dp[0] = 0
    
    for i in range(N):
        new_dp = [INF] * (len_T + 1)
        for j in range(len_T + 1):
            if dp[j] == INF:
                continue
            # Option 1: Do nothing
            if new_dp[j] > dp[j]:
                new_dp[j] = dp[j]
            # Option 2: Choose a string from the current bag
            for s in bags[i]:
                len_s = len(s)
                if j + len_s <= len_T and T[j:j+len_s] == s:
                    if new_dp[j + len_s] > dp[j] + 1:
                        new_dp[j + len_s] = dp[j] + 1
        dp = new_dp
    
    if dp[len_T] == INF:
        print(-1)
    else:
        print(dp[len_T])

if __name__ == "__main__":
    main()