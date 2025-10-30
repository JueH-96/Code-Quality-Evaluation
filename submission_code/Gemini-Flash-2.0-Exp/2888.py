class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        n = len(nums)
        counts = {}
        for num in nums:
            counts[num] = counts.get(num, 0) + 1
        
        dominant = None
        for num, count in counts.items():
            if count * 2 > n:
                dominant = num
                break
        
        left_count = 0
        for i in range(n - 1):
            if nums[i] == dominant:
                left_count += 1
            
            left_len = i + 1
            right_len = n - 1 - i
            
            if left_len > 0:
                if left_count * 2 > left_len:
                    right_count = counts[dominant] - left_count
                    if right_len > 0:
                        if right_count * 2 > right_len:
                            return i
        
        return -1