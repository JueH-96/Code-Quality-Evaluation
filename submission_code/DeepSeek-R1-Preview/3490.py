class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        even_count = 0
        odd_count = 0
        len_end_even = 0
        len_end_odd = 0
        
        for num in nums:
            if num % 2 == 0:
                even_count += 1
                new_len_end_even = max(len_end_even, len_end_odd + 1)
                len_end_even, len_end_odd = new_len_end_even, len_end_odd
            else:
                odd_count += 1
                new_len_end_odd = max(len_end_odd, len_end_even + 1)
                len_end_even, len_end_odd = len_end_even, new_len_end_odd
        
        case1_max = max(even_count, odd_count)
        case2_max = max(len_end_even, len_end_odd)
        
        return max(case1_max, case2_max)