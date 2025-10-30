def solve():
    s = input()
    n = len(s)
    mod = 998244353
    q_count = s.count('?')
    total_strings = pow(52, q_count, mod)
    
    dp = {}

    def count_valid(index, d1, d2, o, s_):
        if index == n:
            return 1
        
        state = (index, d1, d2, o)
        if state in dp:
            return dp[state]

        ans = 0
        
        chars = []
        if s_[index] == '?':
            for i in range(26):
                chars.append(chr(ord('A') + i))
            for i in range(26):
                chars.append(chr(ord('a') + i))
        else:
            chars.append(s_[index])

        for char in chars:
            nd1 = d1
            nd2 = d2
            no = o
            
            if d1 == 0 and 'A' <= char <= 'Z':
                nd1 = 1
            elif d1 == 1 and d2 == 0 and 'A' <= char <= 'Z':
                nd2 = 1
            elif d1 == 1 and d2 == 1 and o == 0 and 'a' <= char <= 'z':
                no = 1
            elif d1 == 1 and d2 == 1 and o == 1 and 'A' <= char <= 'Z' and nd1 == nd2:
                continue
            
            ans = (ans + count_valid(index + 1, nd1, nd2, no, s_)) % mod

        dp[state] = ans
        return ans

    print(count_valid(0, 0, 0, 0, s))

solve()