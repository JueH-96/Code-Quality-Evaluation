import sys

MOD = 998244353

def main() -> None:
    data = sys.stdin.read().strip().split()
    if not data:
        return
    n = int(data[0])
    a = list(map(int, data[1:]))

    # pre-compute 10^d (mod MOD) for 1 ≤ d ≤ 10
    pow10 = [1] * 11              # pow10[d] = 10^d  (mod MOD)
    for d in range(1, 11):
        pow10[d] = (pow10[d - 1] * 10) % MOD

    ans = 0
    prefix_sum = 0                # Σ A_i for i < current position, modulo MOD

    for idx, val in enumerate(a):
        # idx = number of previous elements (0-based index)
        if idx:                   # contribution exists only when there is a previous element
            digits = len(str(val))
            ans = (ans + prefix_sum * pow10[digits]) % MOD      # term A_i * 10^{len(A_j)}
            ans = (ans + idx * val) % MOD                       # term A_j
        prefix_sum = (prefix_sum + val) % MOD                   # update prefix sum

    print(ans % MOD)

if __name__ == "__main__":
    main()