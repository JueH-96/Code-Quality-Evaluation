from math import factorial

class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        # Pre-calculate factorials for 0..n
        fact = [1]*(n+1)
        for i in range(1, n+1):
            fact[i] = fact[i-1]*i

        # Function to compute number of distinct arrangements (permutations) 
        # of an n-digit number with a given digit frequency distribution
        # (without leading zero).
        def count_arrangements(freq):
            # total arrangements if any order of digits (multiset permutation formula)
            tot = fact[n]
            for d in range(10):
                tot //= fact[freq[d]]
            # If there are zeros, subtract those arrangements that start with 0.
            if freq[0] > 0:
                # count arrangements with a fixed leading 0.
                # We remove one zero from frequency.
                rem = fact[n-1]
                rem //= fact[freq[0]-1]
                for d in range(1, 10):
                    rem //= fact[freq[d]]
                tot = tot - rem
            return tot
        
        # We'll generate all palindromic numbers with n digits that satisfy:
        # 1) They have no leading zero.
        # 2) They are palindromic.
        # 3) They are divisible by k.
        #
        # Then for each valid palindrome we get its digit frequency distribution.
        # "Good integers" are counted by all rearrangements of that digit multiset (that don't have leading zero).
        # But multiple palindrome constructions can lead to the same multiset.
        #
        # We use a set (or dictionary) to avoid double counting same multiset.
        good_sets = {}

        # Helper to update dictionary from a palindrome (as string)
        def process_palindrome(s):
            # convert s to integer, skip if it has any leading zero in its own representation (should not happen)
            if s[0] == '0':
                return
            num = int(s)
            if num % k != 0:
                return
            # Compute frequency count for digits 0-9
            freq = [0]*10
            for ch in s:
                freq[int(ch)] += 1
            freq_tuple = tuple(freq)
            if freq_tuple not in good_sets:
                good_sets[freq_tuple] = count_arrangements(freq)
        
        # Generation depends on whether n is odd length or even length
        if n == 1:
            # Single digit palindromes: from 1 to 9.
            for d in range(1,10):
                s = str(d)
                # if s % k == 0 then update good_sets
                if int(s) % k == 0:
                    freq = [0]*10
                    freq[d] = 1
                    freq_tuple = tuple(freq)
                    if freq_tuple not in good_sets:
                        good_sets[freq_tuple] = count_arrangements(freq)
        else:
            if n % 2 == 0:
                half = n // 2
                # left half digits: first digit from 1..9, rest 0..9.
                # We iterate over all possibilities for the left part.
                # We'll use recursion or iterative loops since n<=10 -> half up to 5 digits, at most 9*10^(half-1) iterations.
                def gen_left(curr, length):
                    if length == half:
                        yield curr
                    else:
                        for d in range(10):
                            # For the first digit position, do not allow 0.
                            if length == 0 and d == 0:
                                continue
                            yield from gen_left(curr + str(d), length+1)
                for left in gen_left("", 0):
                    # The palindrome is left + reverse(left)
                    s = left + left[::-1]
                    process_palindrome(s)
            else:
                half = n // 2
                # For odd length, we have left half, and a middle digit.
                def gen_left(curr, length):
                    if length == half:
                        yield curr
                    else:
                        for d in range(10):
                            if length == 0 and d == 0:
                                continue
                            yield from gen_left(curr + str(d), length+1)
                for left in gen_left("", 0):
                    for mid in range(10):
                        s = left + str(mid) + left[::-1]
                        process_palindrome(s)
        
        # Sum over distinct multisets the count of arrangements.
        ans = sum(good_sets.values())
        return ans

# Example usage:
# sol = Solution()
# print(sol.countGoodIntegers(3, 5))  # Output: 27
# print(sol.countGoodIntegers(1, 4))  # Output: 2
# print(sol.countGoodIntegers(5, 6))  # Output: 2468