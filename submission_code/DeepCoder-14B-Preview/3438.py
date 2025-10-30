class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        even_count = 0
        odd_count = 0
        for num in nums:
            if num % 2 == 0:
                even_count += 1
            else:
                odd_count += 1
        
        case1_max = max(even_count, odd_count)
        
        even_dp = 0
        odd_dp = 0
        for num in nums:
            if num % 2 == 0:
                new_even = max(even_dp, odd_dp + 1)
                even_dp = new_even
            else:
                new_odd = max(odd_dp, even_dp + 1)
                odd_dp = new_odd
        
        case2_max = max(even_dp, odd_dp)
        
        return max(case1_max, case2_max)