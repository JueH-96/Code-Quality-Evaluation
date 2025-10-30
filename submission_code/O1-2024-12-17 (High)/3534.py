class Solution:
    def countPairs(self, nums: List[int]) -> int:
        
        def can_become(x: int, y: int) -> bool:
            """
            Return True if x can become y by performing at most one swap of two digits in x.
            Leading zeros are allowed after the swap.
            """
            if x == y:
                return True
            sx, sy = str(x), str(y)
            sx_len = len(sx)
            
            # Try all possible single swaps in x's digits
            sx_list = list(sx)
            for i in range(sx_len):
                for j in range(i+1, sx_len):
                    # Swap
                    sx_list[i], sx_list[j] = sx_list[j], sx_list[i]
                    
                    # Convert to int to ignore leading zeros
                    if int("".join(sx_list)) == y:
                        # Swap back before returning
                        sx_list[i], sx_list[j] = sx_list[j], sx_list[i]
                        return True
                    
                    # Swap back to restore original
                    sx_list[i], sx_list[j] = sx_list[j], sx_list[i]
            
            return False
        
        def is_almost_equal(a: int, b: int) -> bool:
            """
            Two numbers are almost equal if we can make them equal by performing
            at most one swap of two digits in either a or b.
            """
            # We check both directions because we can choose either a or b for the swap
            return can_become(a, b) or can_become(b, a)
        
        count = 0
        n = len(nums)
        
        # Check all pairs (i < j)
        for i in range(n):
            for j in range(i+1, n):
                if is_almost_equal(nums[i], nums[j]):
                    count += 1
        
        return count