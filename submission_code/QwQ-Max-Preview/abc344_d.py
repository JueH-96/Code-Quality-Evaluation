import sys

def main():
    T = sys.stdin.readline().strip()
    n = len(T)
    N = int(sys.stdin.readline())
    bags = []
    for _ in range(N):
        parts = sys.stdin.readline().split()
        A_i = int(parts[0])
        strings = parts[1:]
        bags.append(strings)
    
    INF = float('inf')
    dp = [[INF] * (n + 1) for _ in range(N + 1)]
    dp[0][0] = 0
    
    for i in range(1, N + 1):
        current_bag = bags[i-1]
        for k_prev in range(n + 1):
            if dp[i-1][k_prev] == INF:
                continue
            # Do nothing
            if dp[i][k_prev] > dp[i-1][k_prev]:
                dp[i][k_prev] = dp[i-1][k_prev]
            # Take a string from current bag
            for s in current_bag:
                s_len = len(s)
                new_pos = k_prev + s_len
                if new_pos > n:
                    continue
                if T[k_prev:new_pos] == s:
                    new_cost = dp[i-1][k_prev] + 1
                    if dp[i][new_pos] > new_cost:
                        dp[i][new_pos] = new_cost
    
    result = dp[N][n]
    print(-1 if result == INF else result)

if __name__ == "__main__":
    main()