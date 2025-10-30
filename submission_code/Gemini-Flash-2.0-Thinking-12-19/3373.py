class Solution:
    def maximumPrimeDifference(self, nums: List[int]) -> int:
        def is_prime(n):
            if n <= 1:
                return False
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    return False
            return True

        prime_indices = []
        for i in range(len(nums)):
            if is_prime(nums[i]):
                prime_indices.append(i)

        if len(prime_indices) < 2:
            return 0
        
        max_diff = 0
        for i in range(len(prime_indices)):
            for j in range(len(prime_indices)):
                max_diff = max(max_diff, abs(prime_indices[i] - prime_indices[j]))
        return max_diff