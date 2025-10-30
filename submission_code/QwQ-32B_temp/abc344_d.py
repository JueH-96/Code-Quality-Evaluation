def main():
    import sys
    T = sys.stdin.readline().strip()
    N = int(sys.stdin.readline())
    bags = []
    for _ in range(N):
        parts = sys.stdin.readline().split()
        A_i = int(parts[0])
        strings = parts[1:A_i+1]
        bags.append(strings)
    
    len_T = len(T)
    INF = float('inf')
    dp = [INF] * (len_T + 1)
    dp[0] = 0
    
    for bag in bags:
        temp_dp = dp.copy()
        for k in range(len_T + 1):
            if dp[k] == INF:
                continue
            for s in bag:
                s_len = len(s)
                end = k + s_len
                if end > len_T:
                    continue
                if T[k:end] == s:
                    if temp_dp[end] > dp[k] + 1:
                        temp_dp[end] = dp[k] + 1
        dp = temp_dp
    
    if dp[len_T] == INF:
        print(-1)
    else:
        print(dp[len_T])

if __name__ == "__main__":
    main()