class Solution:
    def canSplitArray(self, nums: List[int], m: int) -> bool:
        n = len(nums)
        
        def can_split(target):
            count = 1
            curr_sum = 0
            for num in nums:
                if curr_sum + num > target:
                    count += 1
                    curr_sum = num
                else:
                    curr_sum += num
            return count <= n
        
        left, right = min(nums), sum(nums)
        while left <= right:
            mid = (left + right) // 2
            if can_split(mid):
                right = mid - 1
            else:
                left = mid + 1
        
        return left == n