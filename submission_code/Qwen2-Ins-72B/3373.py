class Solution:
    def maximumPrimeDifference(self, nums: List[int]) -> int:
        is_prime = [False] * 101
        for i in range(2, 101):
            if not is_prime[i]:
                for j in range(i * i, 101, i):
                    is_prime[j] = True
        prime_indices = [i for i, num in enumerate(nums) if not is_prime[num]]
        return max(prime_indices) - min(prime_indices)