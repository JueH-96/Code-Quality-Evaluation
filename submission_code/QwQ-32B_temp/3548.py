class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        from math import factorial

        # Precompute factorials up to 10! since n is at most 10
        fact = [1] * (11)
        for i in range(1, 11):
            fact[i] = fact[i-1] * i

        def count_permutations(counts):
            total = fact[sum(counts)]
            denom = 1
            for c in counts:
                denom *= fact[c]
            total //= denom

            # Subtract permutations starting with 0
            if counts[0] > 0:
                new_counts = list(counts)
                new_counts[0] -= 1
                remaining = fact[sum(new_counts)]
                for c in new_counts:
                    remaining //= fact[c]
                total -= remaining

            return total if total >= 0 else 0

        m = (n + 1) // 2
        unique_multisets = set()

        # Iterate over all possible first halves
        start = 10 ** (m - 1)
        end = 10 ** m
        for first in range(start, end):
            s = str(first)
            if len(s) != m:
                continue  # Shouldn't happen, but just in case

            # Form the full palindrome
            if n % 2 == 0:
                full = s + s[::-1]
            else:
                full = s + s[:-1][::-1]

            if len(full) != n:
                continue  # Shouldn't happen, but just in case

            # Check leading zero
            if full[0] == '0':
                continue

            num = int(full)
            if num % k != 0:
                continue

            # Compute the multiset counts
            counts = [0] * 10
            for c in full:
                counts[int(c)] += 1
            unique_multisets.add(tuple(counts))

        total = 0
        for counts_tuple in unique_multisets:
            counts = list(counts_tuple)
            cnt = count_permutations(counts)
            total += cnt

        return total