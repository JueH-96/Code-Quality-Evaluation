from collections import Counter

class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        total_count = Counter(nums)
        left_count = Counter()
        dominant_element = None
        min_index = float('inf')
        
        for i, num in enumerate(nums):
            left_count[num] += 1
            if total_count[num] * 2 > len(nums) and (dominant_element is None or left_count[num] > left_count[dominant_element]):
                dominant_element = num
            
            if dominant_element is not None and left_count[dominant_element] * 2 > i + 1 and (total_count[dominant_element] - left_count[dominant_element]) * 2 > len(nums) - i - 1:
                min_index = min(min_index, i)
        
        return min_index if min_index != float('inf') else -1