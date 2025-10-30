class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        if n == 1:
            count = 0
            for i in range(1, 10):
                if i % k == 0:
                    count += 1
            return count

        def is_k_palindromic(digits, k):
            num_str = "".join(map(str, digits))
            if num_str != num_str[::-1]:
                return False
            num = int(num_str)
            return num % k == 0

        def is_good(digits, k):
            import itertools
            for perm in itertools.permutations(digits):
                if perm[0] == 0:
                    continue
                if is_k_palindromic(list(perm), k):
                    return True
            return False

        count = 0
        for i in range(10**(n-1), 10**n):
            digits = list(map(int, str(i)))
            if is_good(digits, k):
                count += 1
        return count