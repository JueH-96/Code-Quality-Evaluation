from math import factorial

class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        # Precompute factorials up to n
        fact = [1] * (n + 1)
        for i in range(1, n + 1):
            fact[i] = fact[i - 1] * i

        # Generate all n-digit palindromes divisible by k
        half_len = (n + 1) // 2
        signatures = set()

        # Helper to build palindrome int from its first half digits
        def build_palindrome(half_digits):
            s = ''.join(str(d) for d in half_digits)
            if n % 2 == 0:
                full = s + s[::-1]
            else:
                full = s + s[-2::-1]
            return int(full), full

        # DFS to enumerate first half
        def dfs(pos, half_digits):
            if pos == half_len:
                val, s = build_palindrome(half_digits)
                if val % k == 0:
                    # compute digit counts signature
                    cnt = [0]*10
                    for ch in s:
                        cnt[ord(ch)-48 if False else int(ch)] += 1
                    signatures.add(tuple(cnt))
                return
            # choose digit
            for d in range(0, 10):
                if pos == 0 and d == 0:
                    continue
                half_digits.append(d)
                dfs(pos+1, half_digits)
                half_digits.pop()

        dfs(0, [])

        # For each signature, count how many n-digit integers (no leading zero)
        # have exactly that multiset of digits
        total = 0
        for sig in signatures:
            # total permutations = n! / prod(sig[d]!)
            denom = 1
            for c in sig:
                denom *= fact[c]
            total_perms = fact[n] // denom

            # subtract those with leading zero
            c0 = sig[0]
            if c0 > 0:
                # fix leading zero, then arrange the rest
                # (n-1)! / ((c0-1)! * prod_{d>0} c_d!)
                denom2 = fact[c0 - 1]
                for d in range(1, 10):
                    denom2 *= fact[sig[d]]
                bad = fact[n-1] // denom2
            else:
                bad = 0

            total += (total_perms - bad)

        return total