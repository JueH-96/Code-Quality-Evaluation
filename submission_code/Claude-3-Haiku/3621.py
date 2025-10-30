class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        n = len(nums)
        if n == 1:
            return 0 if nums[0] == k else -1
        
        # Count the number of elements equal to k
        count_k = nums.count(k)
        if count_k == n:
            return 0
        
        # Find the valid integers
        valid_ints = set()
        for h in range(1, 101):
            is_valid = True
            for num in nums:
                if num > h and num != k:
                    is_valid = False
                    break
            if is_valid:
                valid_ints.add(h)
        
        if not valid_ints:
            return -1
        
        # Find the minimum number of operations
        min_ops = float('inf')
        for h in valid_ints:
            ops = 0
            for num in nums:
                if num > h:
                    ops += num - h
            min_ops = min(min_ops, ops)
        
        return min_ops