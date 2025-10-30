mod = 998244353

def main():
    import sys
    S = sys.stdin.readline().strip()
    n = len(S)
    
    trans_lower = [0] * 55
    for x in range(26):
        s1 = 1 + x
        trans_lower[s1] = s1
        s2 = 1 + 26 + x
        trans_lower[s2] = 53
    trans_lower[0] = 0
    trans_lower[53] = 53
    trans_lower[54] = 54
    
    trans_upper = [[0] * 26 for _ in range(55)]
    for x in range(26):
        trans_upper[0][x] = 1 + x
        for x0 in range(26):
            s1 = 1 + x0
            if x0 == x:
                trans_upper[s1][x] = 1 + 26 + x0
            else:
                trans_upper[s1][x] = 1 + x
        for x0 in range(26):
            s2 = 1 + 26 + x0
            if x0 == x:
                trans_upper[s2][x] = s2
            else:
                trans_upper[s2][x] = 1 + x
        trans_upper[53][x] = 54
        trans_upper[54][x] = 54
        
    dp = [0] * 55
    dp[0] = 1
    
    for c in S:
        new_dp = [0] * 55
        if c == '?':
            for s in range(55):
                cnt = dp[s]
                if cnt == 0:
                    continue
                ns_low = trans_lower[s]
                new_dp[ns_low] = (new_dp[ns_low] + cnt * 26) % mod
                for x in range(26):
                    ns_up = trans_upper[s][x]
                    new_dp[ns_up] = (new_dp[ns_up] + cnt) % mod
        elif c.isupper():
            x = ord(c) - ord('A')
            for s in range(55):
                cnt = dp[s]
                if cnt == 0:
                    continue
                ns = trans_upper[s][x]
                new_dp[ns] = (new_dp[ns] + cnt) % mod
        else:
            for s in range(55):
                cnt = dp[s]
                if cnt == 0:
                    continue
                ns = trans_lower[s]
                new_dp[ns] = (new_dp[ns] + cnt) % mod
        dp = new_dp

    total = 0
    for i in range(54):
        total = (total + dp[i]) % mod
    print(total)

if __name__ == '__main__':
    main()