class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        
        def check(length):
            count = 0
            for i in range(n - length + 1):
                changes = sum(nums[j] != nums[j+1] for j in range(i, i + length - 1))
                if changes <= k:
                    return True
            return False
        
        left, right = 1, n
        while left <= right:
            mid = (left + right) // 2
            if check(mid):
                left = mid + 1
            else:
                right = mid - 1
        
        return right