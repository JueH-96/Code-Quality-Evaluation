class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        n = len(nums)
        counts = {}
        for num in nums:
            counts[num] = counts.get(num, 0) + 1
        
        dominant = 0
        max_count = 0
        for num, count in counts.items():
            if count > max_count:
                max_count = count
                dominant = num
        
        left_count = 0
        for i in range(n - 1):
            if nums[i] == dominant:
                left_count += 1
            
            left_len = i + 1
            right_len = n - left_len
            
            if left_count * 2 > left_len and (max_count - left_count) * 2 > right_len:
                return i
        
        return -1