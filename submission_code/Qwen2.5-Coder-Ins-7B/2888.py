class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        total_count = Counter(nums)
        dominant = nums[0]
        for num, count in total_count.items():
            if count * 2 > len(nums):
                dominant = num
                break
        
        left_count = 0
        for i, num in enumerate(nums):
            if num == dominant:
                left_count += 1
                right_count = total_count[dominant] - left_count
                if left_count * 2 > i + 1 and right_count * 2 > len(nums) - i - 1:
                    return i
        
        return -1