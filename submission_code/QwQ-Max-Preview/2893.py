class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        first_num = nums[0]
        even_max = -float('inf')
        odd_max = -float('inf')
        
        if first_num % 2 == 0:
            even_max = first_num
        else:
            odd_max = first_num
        
        for i in range(1, n):
            num = nums[i]
            if num % 2 == 0:
                temp_even = max(even_max + num, odd_max + num - x)
                new_even = max(even_max, temp_even)
                new_odd = odd_max
            else:
                temp_odd = max(odd_max + num, even_max + num - x)
                new_odd = max(odd_max, temp_odd)
                new_even = even_max
            even_max, odd_max = new_even, new_odd
        
        return max(even_max, odd_max)