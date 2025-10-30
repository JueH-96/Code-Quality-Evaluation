class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        n = len(nums)
        count = {}
        for num in nums:
            count[num] = count.get(num, 0) + 1
        
        dominant = -1
        for num, freq in count.items():
            if freq * 2 > n:
                dominant = num
                break
        
        left_count = 0
        for i in range(n - 1):
            if nums[i] == dominant:
                left_count += 1
            
            right_count = count[dominant] - left_count
            
            if left_count * 2 > (i + 1) and right_count * 2 > (n - i - 1):
                return i
        
        return -1