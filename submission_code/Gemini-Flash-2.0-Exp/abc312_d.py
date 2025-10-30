def solve():
    s = input()
    n = len(s)
    q_count = s.count('?')
    
    dp = {}
    
    def count_ways(index, balance):
        if index == n:
            return 1 if balance == 0 else 0
        
        if (index, balance) in dp:
            return dp[(index, balance)]
        
        ways = 0
        
        if s[index] == '(':
            ways = count_ways(index + 1, balance + 1)
        elif s[index] == ')':
            if balance > 0:
                ways = count_ways(index + 1, balance - 1)
        else:
            ways = count_ways(index + 1, balance + 1)
            if balance > 0:
                ways = (ways + count_ways(index + 1, balance - 1)) % 998244353
        
        dp[(index, balance)] = ways
        return ways
    
    print(count_ways(0, 0) % 998244353)

solve()