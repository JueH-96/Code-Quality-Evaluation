def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    T = input[ptr]
    ptr += 1
    N = int(input[ptr])
    ptr += 1
    
    bags = []
    for _ in range(N):
        A_i = int(input[ptr])
        ptr += 1
        strings = input[ptr:ptr+A_i]
        ptr += A_i
        bags.append(strings)
    
    len_T = len(T)
    INF = float('inf')
    
    # Initialize DP table
    dp = [[INF] * (len_T + 1) for _ in range(N + 1)]
    dp[0][0] = 0
    
    for k in range(1, N + 1):
        current_bag = bags[k - 1]
        for i in range(len_T + 1):
            if dp[k-1][i] == INF:
                continue
            # Option 1: skip this bag
            if dp[k][i] > dp[k-1][i]:
                dp[k][i] = dp[k-1][i]
            # Option 2: take a string from this bag
            for s in current_bag:
                s_len = len(s)
                if i + s_len > len_T:
                    continue
                if T[i:i+s_len] == s:
                    new_i = i + s_len
                    new_cost = dp[k-1][i] + 1
                    if dp[k][new_i] > new_cost:
                        dp[k][new_i] = new_cost
    
    result = dp[N][len_T]
    if result == INF:
        print(-1)
    else:
        print(result)

if __name__ == '__main__':
    main()