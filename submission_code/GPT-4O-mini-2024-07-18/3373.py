class Solution:
    def maximumPrimeDifference(self, nums: List[int]) -> int:
        def is_prime(n):
            if n <= 1:
                return False
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    return False
            return True
        
        prime_indices = [i for i in range(len(nums)) if is_prime(nums[i])]
        
        return prime_indices[-1] - prime_indices[0] if prime_indices else 0