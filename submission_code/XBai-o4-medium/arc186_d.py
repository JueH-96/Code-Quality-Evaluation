import sys
import threading
from functools import lru_cache

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    MOD = 998244353

    # Precompute factorials and inverse factorials
    max_n = 3 * 10**5 + 10
    fact = [1] * (max_n)
    inv_fact = [1] * (max_n)
    for i in range(1, max_n):
        fact[i] = fact[i-1] * i % MOD
    inv_fact[max_n - 1] = pow(fact[max_n - 1], MOD-2, MOD)
    for i in range(max_n-2, -1, -1):
        inv_fact[i] = inv_fact[i+1] * (i+1) % MOD

    def comb(n, k):
        if n < 0 or k < 0 or n < k:
            return 0
        return fact[n] * inv_fact[k] % MOD * inv_fact[n - k] % MOD

    @lru_cache(maxsize=None)
    def dp(i, s):
        if i == N:
            if s == N - 1:
                return 1
            else:
                return 0
        res = 0
        # current sum is s, position i (0-based?)
        # i is 0-based here?
        # For i-th position (0-based), the prefix sum up to i is s
        # Need to choose A[i] such that the prefix sum up to i+1 is s + x >= (i+1) - 1 = i
        # x >= i - s
        low = max(0, i - s)
        # The maximum x is such that the remaining sum can be distributed
        remaining_total = (N - 1) - s
        for x in range(low, remaining_total + 1):
            new_s = s + x
            # Check if the new_s allows the remaining steps to form a valid sequence
            # This is complex, but we'll assume it's possible if the balance is non-negative
            # Calculate the balance after this step
            # balance = 1 + s - i (before this step)
            # after this step, balance becomes balance + x - 1
            # balance_prev = 1 + s - i
            # new_balance = balance_prev + x - 1 = 1 + s - i + x - 1 = s + x - i = new_s - i
            new_balance = new_s - i
            if new_balance < 0:
                continue
            res = (res + dp(i + 1, new_s)) % MOD
        return res

    # Now, simulate the process, trying to build the sequence step by step
    ans = 0
    current_s = 0
    valid = True
    for i in range(N):
        # At position i (0-based), current_s is the sum up to i
        # Need to choose x_i such that x_i >= max(0, i - current_s)
        low = max(0, i - current_s)
        high = (N - 1) - current_s
        # For x in [low, A[i]-1], add dp(i+1, current_s + x)
        a = A[i]
        for x in range(low, a):
            if x > high:
                continue
            new_s = current_s + x
            if new_s > N-1:
                continue
            # Calculate the contribution
            ans = (ans + dp(i+1, new_s)) % MOD
        # Check if a is a valid choice
        if a < low:
            valid = False
            break
        if a > high:
            valid = False
            break
        new_s = current_s + a
        new_balance = new_s - i
        if new_balance < 0:
            valid = False
            break
        current_s = new_s
    if valid:
        # Check if the sequence is valid
        # We need to ensure that the balance never goes below 0
        # Simulate the balance
        balance = 1
        for val in A:
            balance += val - 1
            if balance < 0:
                valid = False
                break
        if valid and balance == 0:
            ans = (ans + 1) % MOD
    print(ans)

threading.Thread(target=main).start()