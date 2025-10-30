class Solution:
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        n = len(nums)
        total_subarrays = n * (n + 1) // 2
        
        term1 = 0
        for left in range(n):
            seen = set()
            for right in range(left, n):
                seen.add(nums[right])
                term1 += len(seen)
        
        def count_subarrays_from_mask(mask):
            total = 0
            i = 0
            while i < n:
                if mask[i] == 1:
                    j = i
                    while j < n and mask[j] == 1:
                        j += 1
                    L = j - i
                    total += L * (L + 1) // 2
                    i = j
                else:
                    i += 1
            return total
        
        term3 = 0
        for x in range(1, n):
            mask_without_x = [1] * n
            mask_without_x1 = [1] * n
            mask_without_both = [1] * n
            
            for i in range(n):
                if nums[i] == x:
                    mask_without_x[i] = 0
                if nums[i] == x + 1:
                    mask_without_x1[i] = 0
                if nums[i] == x or nums[i] == x + 1:
                    mask_without_both[i] = 0
            
            count_without_x = count_subarrays_from_mask(mask_without_x)
            count_without_x1 = count_subarrays_from_mask(mask_without_x1)
            count_without_both = count_subarrays_from_mask(mask_without_both)
            
            count_contain_both = total_subarrays - count_without_x - count_without_x1 + count_without_both
            term3 += count_contain_both
        
        result = term1 - total_subarrays - term3
        return result