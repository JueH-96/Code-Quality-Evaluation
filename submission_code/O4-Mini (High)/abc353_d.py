import sys

def main():
    data = sys.stdin.readline().split()
    N = int(data[0])
    A = list(map(int, sys.stdin.readline().split()))
    MOD = 998244353

    # Precompute 10^k mod MOD for k up to 10 (max digits of A_i is 10)
    pow10 = [1] * 11
    for i in range(1, 11):
        pow10[i] = pow10[i - 1] * 10 % MOD

    prefix_sum = 0
    sum1 = 0  # will accumulate sum of A_i * 10^{len(A_j)} for i<j
    sum2 = 0  # will accumulate sum of A_j for i<j

    for j, Aj in enumerate(A):
        # number of decimal digits of Aj
        lj = len(str(Aj))
        # contribution from all i<j of A_i * 10^{lj}
        sum1 = (sum1 + prefix_sum * pow10[lj]) % MOD
        # contribution from all i<j of A_j
        sum2 = (sum2 + j * Aj) % MOD
        # update prefix sum of A_i
        prefix_sum = (prefix_sum + Aj) % MOD

    ans = (sum1 + sum2) % MOD
    print(ans)

if __name__ == "__main__":
    main()