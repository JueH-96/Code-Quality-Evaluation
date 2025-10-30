MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    
    # Precompute factorial and inverse factorial modulo MOD
    max_n = N
    fact = [1] * (max_n + 1)
    for i in range(1, max_n + 1):
        fact[i] = fact[i-1] * i % MOD
    inv_fact = [1] * (max_n + 1)
    inv_fact[max_n] = pow(fact[max_n], MOD-2, MOD)
    for i in range(max_n-1, -1, -1):
        inv_fact[i] = inv_fact[i+1] * (i+1) % MOD

    def comb(n, k):
        if n < 0 or k < 0 or n < k:
            return 0
        return fact[n] * inv_fact[k] % MOD * inv_fact[n - k] % MOD

    # Check if A is a valid Polish sequence
    stack = []
    dp = [0] * (N + 1)
    dp[0] = 1
    for i in reversed(range(N)):
        if not stack:
            if A[i] == 0:
                dp[i] = 1
            else:
                dp[i] = 0
            continue
        v = stack.pop()
        if v >= len(stack) + (dp[i+1] > 0 and A[i] == v):
            target = 1 + sum(dp[i+1 + j*v_idx] for j in range(v))
            if target != (N - i):
                dp[i] = 0
            else:
                dp[i] = 1 if A[i] >= v else 0
        else:
            dp[i] = 0
        if A[i] != 0:
            stack.append(A[i])
            if dp[i+1] == 0:
                stack = []
                break
            if len(stack) > 0:
                stack.pop()
                stack.append(A[i])
    valid = dp[0] != 0

    if not valid:
        print(0)
        return

    # Compute the answer using dynamic programming
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def count(pos, rem):
        if rem == 0:
            return 1
        if pos >= N:
            return 0
        res = 0
        max_v = A[pos]
        for v in range(0, max_v + 1):
            if v == 0:
                if rem == 1:
                    res = (res + 1) % MOD
                continue
            if rem - 1 < v:
                continue
            c = comb(rem - 1 - 1, v - 1)
            if v == 0:
                ways = 1 if rem == 1 else 0
            else:
                next_rem = rem - 1
                split_pos = pos + 1
                part_ways = 1
                current = split_pos
                for _ in range(v):
                    part_ways = part_ways * count(current, next_rem - (next_rem // v) * (v - 1)) % MOD
                    next_rem -= (next_rem // v) * (v - 1)
                    current += (next_rem // v) + 1
                ways = part_ways
            res = (res + c * ways) % MOD
        return res

    print(count(0, N) % MOD)

if __name__ == '__main__':
    main()