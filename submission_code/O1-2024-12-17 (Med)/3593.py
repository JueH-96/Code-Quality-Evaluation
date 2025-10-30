class Solution:
    def maxScore(self, nums: List[int]) -> int:
        import math
        
        # Edge case: If there's only 1 element, the best score is that element squared
        if len(nums) == 1:
            return nums[0] * nums[0]
        
        # Helper function to compute gcd of a list
        def list_gcd(arr):
            if not arr:
                return 0  # gcd of empty list treated as 0
            g = arr[0]
            for x in arr[1:]:
                g = math.gcd(g, x)
            return g
        
        # Helper function to compute lcm of a list
        def list_lcm(arr):
            if not arr:
                return 0  # lcm of empty list treated as 0
            l = arr[0]
            for x in arr[1:]:
                l = (l * x) // math.gcd(l, x)
            return l
        
        # Calculate the gcd and lcm of the entire array
        gcd_all = list_gcd(nums)
        lcm_all = list_lcm(nums)
        
        # Score without removing any element
        max_score = gcd_all * lcm_all
        
        # Try removing each element and compute the factor score
        n = len(nums)
        for i in range(n):
            sub = nums[:i] + nums[i+1:]
            gcd_sub = list_gcd(sub)
            lcm_sub = list_lcm(sub)
            max_score = max(max_score, gcd_sub * lcm_sub)
        
        return max_score