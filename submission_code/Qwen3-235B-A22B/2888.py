class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        from collections import defaultdict
        
        n = len(nums)
        counts = defaultdict(int)
        
        # Find the dominant element x
        for num in nums:
            counts[num] += 1
        
        x = None
        for key, val in counts.items():
            if val * 2 > n:
                x = key
                break
        
        count_x = counts[x]
        left_x = 0
        
        # Iterate through possible split points
        for i in range(n - 1):
            if nums[i] == x:
                left_x += 1
            left_len = i + 1
            right_len = n - left_len
            if left_x * 2 > left_len and (count_x - left_x) * 2 > right_len:
                return i
        
        return -1