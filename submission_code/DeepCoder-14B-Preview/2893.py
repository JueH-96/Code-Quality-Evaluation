class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        even = float('-inf')
        odd = float('-inf')
        
        first = nums[0]
        if first % 2 == 0:
            even = first
        else:
            odd = first
        
        for num in nums[1:]:
            if num % 2 == 0:
                # Update even state
                new_even = max(even, even + num, (odd + num - x) if odd != float('-inf') else float('-inf'))
                new_odd = odd
            else:
                # Update odd state
                option1 = (even + num - x) if even != float('-inf') else float('-inf')
                option2 = odd + num
                new_odd = max(odd, option1, option2)
                new_even = even
            
            even, odd = new_even, new_odd
        
        return max(even, odd)