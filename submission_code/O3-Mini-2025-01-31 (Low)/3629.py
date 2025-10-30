MOD = 10**9 + 7

class Solution:
    def lengthAfterTransformations(self, s: str, t: int) -> int:
        # The idea: instead of simulating the string transformations, we compute
        # the final length contributed by each initial character.
        #
        # Each transformation acts as:
        #   If char != 'z': it simply becomes the next char (so remains length 1)
        #   If char is 'z': it becomes "ab" (length 2)
        #
        # Notice that for a letter c, the transformation rule only "splits" (i.e. increases
        # the length from 1 to more than 1) when at some transformation step the letter becomes 'z'.
        # For a letter c, let d = ord('z') - ord(c). Then for the first d transformations,
        # c will simply shift to a new letter (without causing a split) because it is not yet 'z'. 
        # When t > d, the letter becomes 'z' at exactly transformation step d and then at the next
        # transformation, the 'z' causes a splitting and so on.
        #
        # So define a function f(c, t):
        #   - if t <= d: f(c, t) = 1  (since it just shifts and never splits)
        #   - if t > d: f(c, t) = h(t - d) where h(x) is defined as the number of characters produced
        #             starting from a 'z' and performing x more transformations.
        #
        # Next, we deduce a recurrence for h(x):
        #   Base: h(0) = 1   (zero transformations leaves the 'z' unchanged)
        #   For any x >= 1:
        #     A 'z' on a transformation becomes "ab". Let’s denote:
        #       f('a', x-1) contribution from 'a'
        #       f('b', x-1) contribution from 'b'
        #     For a letter c != 'z', note that for a letter c which is d steps before 'z' where d > 0,
        #     if x-1 <= d, then f(c, x-1) = 1 (because it will only shift without splitting)
        #     Otherwise, if x-1 > d, f(c,x-1) = h(x-1-d).
        #
        #   For letter 'a': d = ord('z') - ord('a') = 25.
        #     So f('a', x-1) = 1   if (x-1) <= 25, else = h(x-1-25) = h(x-26)
        #   For letter 'b': d = ord('z') - ord('b') = 24.
        #     So f('b', x-1) = 1   if (x-1) <= 24, else = h(x-1-24) = h(x-25)
        #
        #   Therefore, for x >= 1:
        #     h(x) = f('a', x-1) + f('b', x-1)
        #          = (1 if (x-1) <= 25 else h(x-26)) + (1 if (x-1) <= 24 else h(x-25))
        #
        # We compute h(x) using dynamic programming for x from 0 up to t (which is the maximum extra steps
        # we might use when c='z', d = 0, so t-d = t).
        
        # Precompute H[m] = h(m) for m = 0 to t.
        H = [0] * (t + 1)
        H[0] = 1  # Base: no transformation yields the letter as is.
        for m in range(1, t + 1):
            # For a letter 'a' with 25 steps to 'z'
            if m - 1 <= 25:
                part1 = 1
            else:
                part1 = H[m - 26]
            # For a letter 'b' with 24 steps to 'z'
            if m - 1 <= 24:
                part2 = 1
            else:
                part2 = H[m - 25]
            H[m] = (part1 + part2) % MOD
        
        total = 0
        for char in s:
            d = ord('z') - ord(char)
            # If we do not have enough transformations to even reach 'z',
            # the contribution remains 1.
            if t <= d:
                total = (total + 1) % MOD
            else:
                total = (total + H[t - d]) % MOD
        
        return total