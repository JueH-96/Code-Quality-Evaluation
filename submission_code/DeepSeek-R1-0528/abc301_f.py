mod = 998244353
inv26 = pow(26, mod-2, mod)

def main():
    import sys
    S = sys.stdin.readline().strip()
    dp = {(26, 0, 0): 1}
    
    for char in S:
        new_dp = {}
        for state, v in dp.items():
            a, b, c_val = state
            d_val = 26 - a - b - c_val
            if 'A' <= char <= 'Z':
                if a > 0:
                    new_state = (a-1, b+1, c_val)
                    add_val = v * a % mod * inv26 % mod
                    new_dp[new_state] = (new_dp.get(new_state, 0) + add_val) % mod
                if b > 0:
                    new_state = (a, b-1, c_val+1)
                    add_val = v * b % mod * inv26 % mod
                    new_dp[new_state] = (new_dp.get(new_state, 0) + add_val) % mod
                if c_val > 0:
                    new_state = (a, b, c_val)
                    add_val = v * c_val % mod * inv26 % mod
                    new_dp[new_state] = (new_dp.get(new_state, 0) + add_val) % mod
            elif 'a' <= char <= 'z':
                if c_val > 0:
                    new_state = (a, b, c_val-1)
                    add_val = v * c_val % mod * inv26 % mod
                    new_dp[new_state] = (new_dp.get(new_state, 0) + add_val) % mod
                new_state = (a, b, c_val)
                add_val = v * (26 - c_val) % mod * inv26 % mod
                new_dp[new_state] = (new_dp.get(new_state, 0) + add_val) % mod
            else:
                if a > 0:
                    new_state = (a-1, b+1, c_val)
                    add_val = v * a % mod
                    new_dp[new_state] = (new_dp.get(new_state, 0) + add_val) % mod
                if b > 0:
                    new_state = (a, b-1, c_val+1)
                    add_val = v * b % mod
                    new_dp[new_state] = (new_dp.get(new_state, 0) + add_val) % mod
                if c_val > 0:
                    new_state = (a, b, c_val-1)
                    add_val = v * c_val % mod
                    new_dp[new_state] = (new_dp.get(new_state, 0) + add_val) % mod
                new_state = (a, b, c_val)
                add_val = v * 26 % mod
                new_dp[new_state] = (new_dp.get(new_state, 0) + add_val) % mod
        dp = new_dp

    ans = sum(dp.values()) % mod
    print(ans)

if __name__ == '__main__':
    main()