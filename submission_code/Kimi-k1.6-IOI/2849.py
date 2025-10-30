class Solution:
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        total_subarrays_sum = n * (n * n - 1) // 6
        unique_nums = set(nums)
        max_num = max(nums) if nums else 0
        S = 0
        total_subarrays = n * (n + 1) // 2
        
        def count_without(v):
            current = 0
            total = 0
            for num in nums:
                if num == v:
                    total += current * (current + 1) // 2
                    current = 0
                else:
                    current += 1
            total += current * (current + 1) // 2
            return total
        
        def count_without_x_and_x1(x):
            current = 0
            total = 0
            for num in nums:
                if num == x or num == x + 1:
                    total += current * (current + 1) // 2
                    current = 0
                else:
                    current += 1
            total += current * (current + 1) // 2
            return total
        
        for x in range(1, max_num + 1):
            if (x + 1) not in unique_nums:
                continue
            no_x = count_without(x)
            no_x1 = count_without(x + 1)
            no_x_and_x1 = count_without_x_and_x1(x)
            count_x = total_subarrays - no_x - no_x1 + no_x_and_x1
            S += count_x
        
        return total_subarrays_sum - S