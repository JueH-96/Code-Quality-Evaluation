class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        n = len(nums)
        NEG_INF = -10**18
        dp_even = NEG_INF
        dp_odd = NEG_INF
        
        if nums[0] % 2 == 0:
            dp_even = nums[0]
        else:
            dp_odd = nums[0]
            
        for i in range(1, n):
            num = nums[i]
            if num % 2 == 0:
                candidate = dp_even + num
                if dp_odd != NEG_INF:
                    candidate = max(candidate, dp_odd + num - x)
                dp_even = max(dp_even, candidate)
            else:
                candidate = dp_odd + num
                if dp_even != NEG_INF:
                    candidate = max(candidate, dp_even + num - x)
                dp_odd = max(dp_odd, candidate)
                
        return max(dp_even, dp_odd)