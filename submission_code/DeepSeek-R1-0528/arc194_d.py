MOD = 998244353

import sys
sys.setrecursionlimit(100000)

def main():
    data = sys.stdin.read().splitlines()
    n = int(data[0])
    s = data[1].strip()
    
    max_components = n // 2 + 1
    fact = [1] * (max_components + 1)
    for i in range(1, max_components + 1):
        fact[i] = fact[i - 1] * i % MOD

    dp = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            if i > j:
                dp[i][j] = 1
    
    for length in range(2, n + 1, 2):
        for l in range(0, n - length + 1):
            r = l + length - 1
            stack = []
            components = []
            start = l
            valid = True
            for i in range(l, r + 1):
                if s[i] == '(':
                    stack.append(i)
                else:
                    if stack:
                        stack.pop()
                        if not stack:
                            components.append((start, i))
                            start = i + 1
                    else:
                        valid = False
                        break
            if not valid or stack:
                continue
            
            product = 1
            for (i, j) in components:
                if j == i + 1:
                    inside = 1
                else:
                    inside = dp[i + 1][j - 1]
                product = (product * inside) % MOD
            
            k = len(components)
            dp[l][r] = fact[k] * product % MOD

    print(dp[0][n - 1])

if __name__ == '__main__':
    main()