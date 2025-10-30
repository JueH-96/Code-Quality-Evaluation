import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    A = [int(next(it)) for _ in range(n)]
    MOD = 998244353

    # Precompute powers of 10 up to length 10 (since A_i <= 10^9 has at most 10 digits)
    max_len = 10
    pow10 = [1] * (max_len + 1)
    for i in range(1, max_len + 1):
        pow10[i] = (pow10[i-1] * 10) % MOD

    # Helper to compute number of decimal digits
    def num_digits(x):
        # x > 0
        # faster than str(x) for many calls
        if x < 10:
            return 1
        if x < 100:
            return 2
        if x < 1000:
            return 3
        if x < 10000:
            return 4
        if x < 100000:
            return 5
        if x < 1000000:
            return 6
        if x < 10000000:
            return 7
        if x < 100000000:
            return 8
        if x < 1000000000:
            return 9
        return 10

    ans = 0
    prefix_sum = 0  # sum A[0..j-1] mod MOD

    # We'll build answer by iterating j from 0 to n-1 (1-based j = idx+1)
    # For each j we add:
    #   S1 contribution: prefix_sum * 10^{len(A[j])}
    #   S2 contribution: A[j] * (j)        since there are j elements before index j (0-based)
    for idx, val in enumerate(A):
        l = num_digits(val)
        # S1 part: sum_{i<j} A[i] * 10^{len(A[j])}
        ans = (ans + prefix_sum * pow10[l]) % MOD
        # S2 part: sum_{i<j} A[j]
        ans = (ans + val * idx) % MOD

        # update prefix sum
        prefix_sum = (prefix_sum + val) % MOD

    print(ans)

if __name__ == "__main__":
    main()