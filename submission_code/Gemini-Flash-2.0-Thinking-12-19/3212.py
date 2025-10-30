from collections import defaultdict

class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        first_occurrence = {}
        last_occurrence = {}
        unique_numbers = set()
        for i in range(n):
            num = nums[i]
            unique_numbers.add(num)
            if num not in first_occurrence:
                first_occurrence[num] = i
            last_occurrence[num] = i
        
        allowed_cut_positions = set()
        for i in range(n - 1):
            is_allowed_cut = True
            for num in unique_numbers:
                if not (last_occurrence[num] <= i or first_occurrence[num] >= i + 1):
                    is_allowed_cut = False
                    break
            if is_allowed_cut:
                allowed_cut_positions.add(i)
                
        num_allowed_cuts = len(allowed_cut_positions)
        
        def power(base, exp):
            res = 1
            base %= 10**9 + 7
            while exp > 0:
                if exp % 2 == 1:
                    res = (res * base) % (10**9 + 7)
                exp >>= 1
                base = (base * base) % (10**9 + 7)
            return res
            
        return power(2, num_allowed_cuts)