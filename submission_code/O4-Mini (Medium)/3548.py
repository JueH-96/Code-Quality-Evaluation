from math import factorial

class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        # Precompute factorials up to n
        fact = [1] * (n+1)
        for i in range(1, n+1):
            fact[i] = fact[i-1] * i

        ans = 0

        # Generate all digit-count vectors c[0..9] summing to n
        c = [0]*10
        def gen_counts(pos, remaining):
            nonlocal ans
            if pos == 9:
                # Last digit gets all remaining
                c[9] = remaining
                process_multiset(c)
                return
            # Try all possible counts for digit 'pos'
            for cnt in range(remaining+1):
                c[pos] = cnt
                gen_counts(pos+1, remaining-cnt)
            c[pos] = 0

        def process_multiset(c):
            nonlocal ans
            # Check palindrome-possible parity
            odd_counts = sum((ci & 1) for ci in c)
            if n & 1:
                # odd length: exactly one odd
                if odd_counts != 1:
                    return
            else:
                # even length: no odd
                if odd_counts != 0:
                    return

            # Build half-counts h[i] = c[i]//2
            h = [ci//2 for ci in c]
            L = sum(h)
            # Identify middle digit if odd
            mid_digit = -1
            if n & 1:
                for d in range(10):
                    if c[d] & 1:
                        mid_digit = d
                        break

            # Try to find at least one palindrome permutation divisible by k
            found = False
            perm = [0]*L

            def dfs(pos):
                nonlocal found
                if found:
                    return
                if pos == L:
                    # Check first digit non-zero
                    if L > 0:
                        # Leading digit is perm[0], and we disallowed 0 there already
                        pass
                    else:
                        # L==0 -> only middle digit forms the number
                        if mid_digit == 0:
                            return
                    # Compute modulo k of the full palindrome
                    m = 0
                    # left half
                    for d in perm:
                        m = (m*10 + d) % k
                    # middle
                    if n & 1:
                        m = (m*10 + mid_digit) % k
                    # right half reversed
                    for d in reversed(perm):
                        m = (m*10 + d) % k
                    if m == 0:
                        found = True
                    return

                # Choose next digit for half
                for d in range(10):
                    if h[d] == 0:
                        continue
                    # Leading digit check
                    if pos == 0 and L > 0 and d == 0:
                        continue
                    # Use digit d
                    h[d] -= 1
                    perm[pos] = d
                    dfs(pos+1)
                    h[d] += 1
                    if found:
                        return

            dfs(0)
            if not found:
                return

            # This multiset is good: count all n-digit numbers (no leading zero)
            # Total permutations: n! / prod(c[i]!)
            tot = fact[n]
            for ci in c:
                tot //= fact[ci]
            # Subtract those with leading zero:
            invalid = 0
            if c[0] > 0:
                invalid = fact[n-1]
                # use one zero in front
                invalid //= fact[c[0]-1]
                for d in range(1,10):
                    invalid //= fact[c[d]]
            ans += tot - invalid

        # Start recursion
        gen_counts(0, n)
        return ans