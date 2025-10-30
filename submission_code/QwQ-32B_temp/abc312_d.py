MOD = 998244353

s = input().strip()
n = len(s)

if n % 2 != 0:
    print(0)
else:
    prev = [0] * (n + 1)
    prev[0] = 1
    for i in range(1, n + 1):
        current = [0] * (n + 1)
        c = s[i-1]
        for b in range(n + 1):
            if prev[b] == 0:
                continue
            if c == '(':
                new_b = b + 1
                if new_b <= n:
                    current[new_b] = (current[new_b] + prev[b]) % MOD
            elif c == ')':
                if b >= 1:
                    new_b = b - 1
                    current[new_b] = (current[new_b] + prev[b]) % MOD
            else:  # '?'
                # Option 1: '('
                new_b1 = b + 1
                if new_b1 <= n:
                    current[new_b1] = (current[new_b1] + prev[b]) % MOD
                # Option 2: ')'
                if b >= 1:
                    new_b2 = b - 1
                    current[new_b2] = (current[new_b2] + prev[b]) % MOD
        prev = current[:]
    print(prev[0] % MOD)