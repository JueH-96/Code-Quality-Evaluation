def main():
    import sys
    input = sys.stdin.read().splitlines()
    T = input[0].strip()
    N = int(input[1])
    bags = []
    for i in range(2, 2 + N):
        parts = input[i].split()
        A_i = int(parts[0])
        strs = parts[1:]
        bags.append(strs)
    
    L = len(T)
    INF = float('inf')
    dp = [INF] * (L + 1)
    dp[0] = 0
    
    for bag in bags:
        temp_dp = list(dp)
        for l in range(L + 1):
            if dp[l] == INF:
                continue
            for s in bag:
                k = len(s)
                new_l = l + k
                if new_l > L:
                    continue
                if T[l:new_l] == s:
                    if temp_dp[new_l] > dp[l] + 1:
                        temp_dp[new_l] = dp[l] + 1
        dp = temp_dp
    
    ans = dp[L] if dp[L] != INF else -1
    print(ans)

if __name__ == '__main__':
    main()