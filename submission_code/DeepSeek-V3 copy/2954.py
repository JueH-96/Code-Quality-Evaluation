from collections import defaultdict

class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        n = len(nums)
        if n < k:
            return 0
        
        max_sum = 0
        current_sum = sum(nums[:k])
        freq = defaultdict(int)
        for num in nums[:k]:
            freq[num] += 1
        
        if len(freq) >= m:
            max_sum = current_sum
        
        for i in range(k, n):
            current_sum += nums[i] - nums[i - k]
            freq[nums[i]] += 1
            freq[nums[i - k]] -= 1
            if freq[nums[i - k]] == 0:
                del freq[nums[i - k]]
            if len(freq) >= m:
                max_sum = max(max_sum, current_sum)
        
        return max_sum