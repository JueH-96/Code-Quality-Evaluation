MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    K = int(input[1])
    S = input[2]

    from collections import defaultdict

    dp = [defaultdict(int) for _ in range(N + 1)]
    dp[0][ (0, 0, 0) ] = 1

    for i in range(1, N + 1):
        c = S[i-1]
        for (a_prev, b_prev, c_prev), ways in dp[i-1].items():
            if c == '?':
                chars = ['A', 'B', 'C']
            else:
                chars = [c]
            for char in chars:
                a_new = a_prev + (1 if char == 'A' else 0)
                b_new = b_prev + (1 if char == 'B' else 0)
                c_new = c_prev + (1 if char == 'C' else 0)
                key = (a_new, b_new, c_new)
                dp[i][key] = (dp[i][key] + ways) % MOD

    total = 0
    for i in range(1, N + 1):
        for (a, b, c), ways in dp[i].items():
            if (a - b) % 2 != 0 or (a - c) % 2 != 0 or (b - c) % 2 != 0:
                continue
            y = a // 2 if a >= b and a >= c else b // 2 if b >= a and b >= c else c // 2
            if a >= y and b >= y and c >= y:
                total = (total + ways) % MOD

    print(total % MOD)

if __name__ == '__main__':
    main()