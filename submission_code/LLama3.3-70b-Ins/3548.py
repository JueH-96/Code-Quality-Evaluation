from itertools import permutations

class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        count = 0
        for num in range(10**(n-1), 10**n):
            # Generate all permutations of the digits
            perms = set(int(''.join(p)) for p in permutations(str(num)))
            # Check if any permutation is a palindrome and divisible by k
            for perm in perms:
                if str(perm) == str(perm)[::-1] and perm % k == 0:
                    count += 1
                    break
        return count