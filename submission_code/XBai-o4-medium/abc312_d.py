MOD = 998244353

def main():
    s = input().strip()
    n = len(s)
    max_balance = n
    prev = [0] * (max_balance + 2)  # indexes 0..n+1
    prev[0] = 1

    for i in range(n):
        current = [0] * (max_balance + 2)
        c = s[i]
        for b in range(0, max_balance + 1):
            if prev[b] == 0:
                continue
            if c == '(':
                new_b = b + 1
                if new_b <= max_balance:
                    current[new_b] = (current[new_b] + prev[b]) % MOD
            elif c == ')':
                new_b = b - 1
                if new_b >= 0:
                    current[new_b] = (current[new_b] + prev[b]) % MOD
            else:
                # handle both cases for '?'
                # add '(' case
                new_b1 = b + 1
                if new_b1 <= max_balance:
                    current[new_b1] = (current[new_b1] + prev[b]) % MOD
                # add ')' case
                new_b2 = b - 1
                if new_b2 >= 0:
                    current[new_b2] = (current[new_b2] + prev[b]) % MOD
        prev = current

    print(prev[0] % MOD)

if __name__ == "__main__":
    main()