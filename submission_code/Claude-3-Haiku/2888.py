class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        n = len(nums)
        count = {}
        for num in nums:
            count[num] = count.get(num, 0) + 1
        
        dominant = None
        for num, freq in count.items():
            if freq * 2 > n:
                dominant = num
                break
        
        if dominant is None:
            return -1
        
        for i in range(n-1):
            left_count = 0
            right_count = 0
            for j in range(i+1):
                if nums[j] == dominant:
                    left_count += 1
            for j in range(i+1, n):
                if nums[j] == dominant:
                    right_count += 1
            if left_count * 2 > i+1 and right_count * 2 > n-i-1:
                return i
        
        return -1