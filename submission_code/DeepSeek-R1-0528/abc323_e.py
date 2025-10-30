mod = 998244353
import sys
from collections import defaultdict

def main():
    data = sys.stdin.read().split()
    if not data: 
        return
    N = int(data[0])
    X = int(data[1])
    T_list = list(map(int, data[2:2+N]))

    T1 = T_list[0]
    A = max(0, X - T1 + 1)

    freq_dict = defaultdict(int)
    for t in T_list:
        freq_dict[t] += 1

    invN = pow(N, mod-2, mod)

    dp = [0] * (X+1)
    if X >= 0:
        dp[0] = 1

    for s in range(1, X+1):
        total_val = 0
        for d, cnt in freq_dict.items():
            if d <= s:
                total_val = (total_val + cnt * dp[s-d]) % mod
        dp[s] = total_val * invN % mod

    total_sum = 0
    for s in range(A, X+1):
        total_sum = (total_sum + dp[s]) % mod

    ans = total_sum * invN % mod
    print(ans)

if __name__ == "__main__":
    main()