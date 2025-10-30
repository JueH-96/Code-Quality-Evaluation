class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        n = len(nums)
        prefix_sum = [0] * (n + 1)
        count = {0: 1}
        result = 0

        for i in range(n):
            prefix_sum[i + 1] = (prefix_sum[i] + (1 if nums[i] % modulo == k else 0)) % modulo
            
            target = (prefix_sum[i + 1] - k + modulo) % modulo
            if target in count:
                result += count[target]
            
            count[prefix_sum[i + 1]] = count.get(prefix_sum[i + 1], 0) + 1

        return result