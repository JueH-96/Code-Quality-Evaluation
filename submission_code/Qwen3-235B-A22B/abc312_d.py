MOD = 998244353

def main():
    import sys
    S = sys.stdin.readline().strip()
    n = len(S)
    prev = [0] * (n + 2)
    prev[0] = 1

    for c in S:
        curr = [0] * (n + 2)
        for b in range(n + 1):
            if prev[b] == 0:
                continue
            if c == '(' or c == '?':
                new_b = b + 1
                if new_b <= n:
                    curr[new_b] = (curr[new_b] + prev[b]) % MOD
            if c == ')' or c == '?':
                new_b = b - 1
                if new_b >= 0:
                    curr[new_b] = (curr[new_b] + prev[b]) % MOD
        prev = curr

    print(prev[0] % MOD)

if __name__ == "__main__":
    main()