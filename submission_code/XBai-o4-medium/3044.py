class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        last_occurrence = {}
        n = len(nums)
        for i in range(n):
            num = nums[i]
            last_occurrence[num] = i  # Overwrite to store last occurrence
        
        max_ops = 0
        for x in range(1, k + 1):
            index = last_occurrence[x]
            current_ops = n - index
            if current_ops > max_ops:
                max_ops = current_ops
        return max_ops