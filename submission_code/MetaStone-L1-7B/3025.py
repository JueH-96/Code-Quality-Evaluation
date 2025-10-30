class Solution:
    def min_operations(self, nums: List[int], target: int) -> int:
        sum_total = sum(nums)
        if sum_total < target:
            return -1
        
        needed = [0] * 60
        for k in range(60):
            needed[k] = (target >> k) & 1
        
        available = [0] * 60
        for num in nums:
            k = 0
            while num > 0:
                k += 1
                if num >= (1 << k):
                    break
                num >>= 1
            available[k] += 1
        
        splits = 0
        total_ops = 0
        
        for k in range(59, -1, -1):
            required = needed[k]
            current_available = available[k] + 2 * splits
            if current_available < required:
                delta = required - current_available
                splits_needed = (delta + 1) // 2
                total_ops += splits_needed
                available[k] += 2 * splits_needed
                splits = splits_needed
            else:
                available[k] = current_available
        
        return total_ops