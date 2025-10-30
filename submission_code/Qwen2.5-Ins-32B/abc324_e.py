import sys

def solve():
    input = sys.stdin.read
    data = input().split()
    n = int(data[0])
    t = data[1]
    strings = data[2:]
    
    def preprocess(s):
        dp = [0] * len(s)
        j = 0
        for i in range(len(s)):
            while j < len(t) and i < len(s) and s[i] != t[j]:
                i += 1
            if i < len(s):
                dp[i] = j + 1
                j += 1
        return dp
    
    prefix = [preprocess(s) for s in strings]
    suffix = [preprocess(s[::-1])[::-1] for s in strings]
    
    count = [0] * (len(t) + 1)
    for s in suffix:
        count[s[-1]] += 1
    
    result = 0
    for p in prefix:
        for i in range(len(t) + 1):
            if p[-1] + i >= len(t):
                result += count[i]
        count[p[-1]] += 1
    
    print(result)

solve()