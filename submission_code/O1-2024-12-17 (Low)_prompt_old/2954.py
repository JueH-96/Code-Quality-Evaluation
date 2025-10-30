class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        from collections import defaultdict
        
        freq = defaultdict(int)
        current_sum = 0
        distinct_count = 0
        max_sum = 0
        start = 0
        
        for end in range(len(nums)):
            freq[nums[end]] += 1
            if freq[nums[end]] == 1:
                distinct_count += 1
            current_sum += nums[end]
            
            # Ensure the window size does not exceed k
            if end - start + 1 > k:
                freq[nums[start]] -= 1
                if freq[nums[start]] == 0:
                    distinct_count -= 1
                current_sum -= nums[start]
                start += 1
            
            # When window size is exactly k
            if end - start + 1 == k:
                if distinct_count >= m:
                    max_sum = max(max_sum, current_sum)
        
        return max_sum