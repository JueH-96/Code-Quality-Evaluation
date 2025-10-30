class Solution:
    def minAnagramLength(self, s: str) -> int:
        import math

        n = len(s)
        if n == 0:
            return 0

        # Initialize counts for each character
        counts = [0] * 26
        for c in s:
            counts[ord(c) - 97] += 1

        # Precompute prefix sums for each character
        pre = [[0] * (n + 1) for _ in range(26)]
        for i in range(n):
            current_char = ord(s[i]) - 97
            for j in range(26):
                pre[j][i + 1] = pre[j][i]
            pre[current_char][i + 1] += 1

        # Generate all divisors of n
        divisors = set()
        for i in range(1, int(math.isqrt(n)) + 1):
            if n % i == 0:
                divisors.add(i)
                divisors.add(n // i)
        sorted_divisors = sorted(divisors)

        # Check each possible k in increasing order
        for k in sorted_divisors:
            m = n // k
            # Check if all counts are divisible by m
            valid = True
            for j in range(26):
                if counts[j] % m != 0:
                    valid = False
                    break
            if not valid:
                continue

            # Compute target counts
            target = [counts[j] // m for j in range(26)]

            # Check each block
            ok = True
            for block in range(m):
                start = block * k
                end = start + k
                for j in range(26):
                    if (pre[j][end] - pre[j][start]) != target[j]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return k

        return n  # Fallback, though problem ensures a valid solution exists