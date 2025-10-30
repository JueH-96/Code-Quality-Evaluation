import math

class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        # Precompute factorials up to n
        fact = [1] * (n+1)
        for i in range(1, n+1):
            fact[i] = fact[i-1] * i
        
        # Given a frequency tuple (length 10) of digits, count the number of distinct n-digit permutations
        # that have a nonzero first digit.
        def count_permutations(freq):
            # Total permutations = n! / (prod(f[d]!))
            total = fact[n]
            for f in freq:
                total //= fact[f]
            # Count permutations with leading zero, if any zero present
            if freq[0] == 0:
                return total
            # Remove one 0 and arrange the remaining n-1 digits
            perm_with_leading_zero = fact[n-1]
            # Adjust counts: one fewer for digit 0.
            freq0 = freq[0] - 1
            perm_with_leading_zero //= fact[freq0]
            for d in range(1, 10):
                perm_with_leading_zero //= fact[freq[d]]
            return total - perm_with_leading_zero
        
        # Helper: return frequency tuple for a given number string.
        def freq_tuple(s):
            counts = [0] * 10
            for ch in s:
                counts[ord(ch)-ord('0')] += 1
            return tuple(counts)
        
        good_multisets = set()
        
        # We'll generate palindromic numbers with n digits that are divisible by k.
        # They must have no leading zero.
        # Case 1: n is even
        if n % 2 == 0:
            half_len = n // 2
            # The first half must be a string of half_len digits with no leading zero.
            lower = 10 ** (half_len - 1)
            upper = 10 ** half_len
            for half in range(lower, upper):
                s = str(half).zfill(half_len)  # although half>=lower so already has no leading zero.
                pal = s + s[::-1]
                num = int(pal)
                if num % k == 0:
                    # It's a valid k-palindrome. Record its digit multiset.
                    good_multisets.add(freq_tuple(pal))
        else:
            # n is odd
            half_len = n // 2
            if n == 1:
                # simply one digit numbers; must be nonzero.
                for d in range(1, 10):
                    if d % k == 0:
                        good_multisets.add(freq_tuple(str(d)))
            else:
                # For odd n, construct half and then choose a center digit.
                lower = 10 ** (half_len - 1)
                upper = 10 ** half_len
                for half in range(lower, upper):
                    s = str(half).zfill(half_len)
                    for center in range(10):
                        pal = s + str(center) + s[::-1]
                        num = int(pal)
                        if num % k == 0:
                            good_multisets.add(freq_tuple(pal))
        
        # Sum over all "good" multisets the number of n-digit permutations that yield that multiset.
        total_good = 0
        for ms in good_multisets:
            total_good += count_permutations(ms)
        return total_good


# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.countGoodIntegers(3, 5))  # Expected output: 27
    print(sol.countGoodIntegers(1, 4))  # Expected output: 2
    print(sol.countGoodIntegers(5, 6))  # Expected output: 2468