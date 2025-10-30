class Solution:
    def minSizeSubarray(self, nums: List[int], target: int) -> int:
        n = len(nums)
        total = sum(nums)
        k = target // total
        r = target % total
        
        if r == 0:
            return k * n
        
        arr = nums + nums
        min_len = float('inf')
        left = 0
        current = 0
        
        for right in range(2 * n):
            current += arr[right]
            while current > r and left <= right:
                current -= arr[left]
                left += 1
            if current == r:
                min_len = min(min_len, right - left + 1)
        
        if min_len == float('inf'):
            return -1
        else:
            return k * n + min_len