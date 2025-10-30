class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        # Find the dominant element x using Boyer-Moore majority vote algorithm
        candidate = None
        count = 0
        for num in nums:
            if count == 0:
                candidate = num
                count = 1
            elif num == candidate:
                count += 1
            else:
                count -= 1
        x = candidate
        
        # Count total occurrences of x in nums
        total_count = 0
        for num in nums:
            if num == x:
                total_count += 1
        
        # Now, iterate i from 0 to n-2
        n = len(nums)
        left_count = 0
        for i in range(n - 1):
            if nums[i] == x:
                left_count += 1
            left_length = i + 1
            right_count = total_count - left_count
            right_length = n - left_length
            if left_count * 2 > left_length and right_count * 2 > right_length:
                return i
        return -1