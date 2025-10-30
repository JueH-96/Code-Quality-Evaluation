class Solution:
    def largestInteger(self, nums: List[int], k: int) -> int:
        from collections import defaultdict
        
        element_subarrays = defaultdict(int)
        n = len(nums)
        
        for i in range(n - k + 1):
            subarray = nums[i:i+k]
            unique_elements = set(subarray)
            for num in unique_elements:
                element_subarrays[num] += 1
        
        candidates = [num for num in element_subarrays if element_subarrays[num] == 1]
        
        if not candidates:
            return -1
        else:
            return max(candidates)