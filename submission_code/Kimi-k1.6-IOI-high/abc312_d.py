def main():
    import sys
    MOD = 998244353
    S = sys.stdin.readline().strip()
    n = len(S)
    if n % 2 != 0:
        print(0)
        return
    
    max_balance = n  # Maximum possible balance after n steps
    prev = [0] * (max_balance + 2)
    prev[0] = 1
    
    for i in range(n):
        curr = [0] * (max_balance + 2)
        char = S[i]
        for j in range(max_balance + 1):
            if prev[j] == 0:
                continue
            if char == '(':
                if j + 1 <= max_balance:
                    curr[j + 1] = (curr[j + 1] + prev[j]) % MOD
            elif char == ')':
                if j >= 1:
                    curr[j - 1] = (curr[j - 1] + prev[j]) % MOD
            else:  # '?'
                if j + 1 <= max_balance:
                    curr[j + 1] = (curr[j + 1] + prev[j]) % MOD
                if j >= 1:
                    curr[j - 1] = (curr[j - 1] + prev[j]) % MOD
        prev = curr
    
    print(prev[0] % MOD)

if __name__ == "__main__":
    main()