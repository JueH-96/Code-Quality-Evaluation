class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        # Precompute factorials up to n
        fact = [1] * (n + 1)
        for i in range(1, n + 1):
            fact[i] = fact[i - 1] * i
        # Precompute powers of 10 modulo k
        pow10_mod = [1] * (n + 1)
        for i in range(1, n + 1):
            pow10_mod[i] = (pow10_mod[i - 1] * 10) % k

        m = n // 2
        ans = 0
        half_cnt = [0] * 10

        def process_half_cnt(hcnt):
            nonlocal ans
            # Build the set of (S1+S2)%k for all valid half-permutations
            if m == 0:
                hmods = {0}
            else:
                hmods = set()
                used = list(hcnt)
                def back(i, S1, S2):
                    if i == m:
                        hmods.add((S1 + S2) % k)
                        return
                    for d in range(10):
                        if used[d] > 0:
                            # The first digit of the full palindrome must not be zero
                            if i == 0 and d == 0:
                                continue
                            used[d] -= 1
                            newS1 = (S1 + d * pow10_mod[n - 1 - i]) % k
                            newS2 = (S2 + d * pow10_mod[i]) % k
                            back(i + 1, newS1, newS2)
                            used[d] += 1
                back(0, 0, 0)
            if not hmods:
                return

            # Even-length palindromes
            if n % 2 == 0:
                # We need some half that gives 0 mod k
                if 0 not in hmods:
                    return
                # Build full digit counts
                cnt = [2 * hcnt[d] for d in range(10)]
                # Count all permutations of these n digits without leading zero
                total = fact[n]
                for d in range(10):
                    total //= fact[cnt[d]]
                lead0 = 0
                if cnt[0] > 0:
                    # Count those with a leading zero
                    lead0 = fact[n - 1] // fact[cnt[0] - 1]
                    for d in range(1, 10):
                        lead0 //= fact[cnt[d]]
                ans += total - lead0

            # Odd-length palindromes
            else:
                ten_pow_mid = pow10_mod[m]
                for j in range(10):
                    # Check if there is a half-mod that makes the full palindrome divisible
                    if (-j * ten_pow_mid) % k in hmods:
                        cnt = [2 * hcnt[d] for d in range(10)]
                        cnt[j] += 1
                        total = fact[n]
                        for d in range(10):
                            total //= fact[cnt[d]]
                        lead0 = 0
                        if cnt[0] > 0:
                            lead0 = fact[n - 1] // fact[cnt[0] - 1]
                            for d in range(1, 10):
                                lead0 //= fact[cnt[d]]
                        ans += total - lead0

        # Enumerate all half-count vectors summing to m
        def dfs(pos, rem):
            if pos == 9:
                half_cnt[pos] = rem
                process_half_cnt(tuple(half_cnt))
                return
            for c in range(rem + 1):
                half_cnt[pos] = c
                dfs(pos + 1, rem - c)

        dfs(0, m)
        return ans