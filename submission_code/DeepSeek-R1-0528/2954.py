from collections import defaultdict

class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        n = len(nums)
        current_sum = sum(nums[:k])
        freq = defaultdict(int)
        for i in range(k):
            freq[nums[i]] += 1
        
        max_sum = current_sum if len(freq) >= m else 0
        
        left = 0
        for right in range(k, n):
            freq[nums[left]] -= 1
            if freq[nums[left]] == 0:
                del freq[nums[left]]
            current_sum -= nums[left]
            left += 1
            
            freq[nums[right]] += 1
            current_sum += nums[right]
            
            if len(freq) >= m:
                if current_sum > max_sum:
                    max_sum = current_sum
        
        return max_sum