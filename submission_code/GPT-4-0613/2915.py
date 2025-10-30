class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        n = len(nums)
        prefix = [0] * (n + 1)
        count = [0] * modulo
        count[0] = 1
        interesting_subarrays = 0

        for i in range(n):
            prefix[i + 1] = (prefix[i] + (nums[i] % modulo == k)) % modulo
            interesting_subarrays += count[prefix[i + 1]]
            count[prefix[i + 1]] += 1

        return interesting_subarrays