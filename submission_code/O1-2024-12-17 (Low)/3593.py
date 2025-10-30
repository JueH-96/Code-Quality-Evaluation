class Solution:
    def maxScore(self, nums: List[int]) -> int:
        import math
        
        # Helper function to compute the GCD of a list
        def gcd_list(arr):
            g = arr[0]
            for x in arr[1:]:
                g = math.gcd(g, x)
            return g
        
        # Helper function to compute the LCM of a list
        def lcm_list(arr):
            l = arr[0]
            for x in arr[1:]:
                l = (l * x) // math.gcd(l, x)
            return l
        
        n = len(nums)
        # If there's only one element, return its square (or 0 if we remove it)
        if n == 1:
            return nums[0] * nums[0]  # removing it yields 0, so keep it
        
        # Compute factor score without removal
        current_gcd = gcd_list(nums)
        current_lcm = lcm_list(nums)
        max_score = current_gcd * current_lcm
        
        # Compute factor score after removing each element
        for i in range(n):
            sub_arr = nums[:i] + nums[i+1:]
            if len(sub_arr) == 0:
                score = 0
            else:
                g = gcd_list(sub_arr)
                l = lcm_list(sub_arr)
                score = g * l
            max_score = max(max_score, score)
        
        return max_score