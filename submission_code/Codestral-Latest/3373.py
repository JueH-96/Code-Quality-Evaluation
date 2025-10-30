class Solution:
    def maximumPrimeDifference(self, nums: List[int]) -> int:
        def is_prime(n):
            if n <= 1:
                return False
            if n <= 3:
                return True
            if n % 2 == 0 or n % 3 == 0:
                return False
            i = 5
            while i * i <= n:
                if n % i == 0 or n % (i + 2) == 0:
                    return False
                i += 6
            return True

        prime_indices = [i for i, num in enumerate(nums) if is_prime(num)]

        if len(prime_indices) == 1:
            return 0

        max_diff = 0
        for i in range(len(prime_indices)):
            for j in range(i, len(prime_indices)):
                max_diff = max(max_diff, abs(prime_indices[j] - prime_indices[i]))

        return max_diff