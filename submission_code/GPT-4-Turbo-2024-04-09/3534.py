class Solution:
    def countPairs(self, nums: List[int]) -> int:
        from collections import defaultdict
        
        def can_be_almost_equal(x, y):
            # Convert numbers to string to easily swap characters
            sx, sy = str(x), str(y)
            if sx == sy:
                return True
            if len(sx) != len(sy):
                return False
            
            # Find the positions where the digits differ
            diff = []
            for i in range(len(sx)):
                if sx[i] != sy[i]:
                    diff.append(i)
                    if len(diff) > 2:
                        return False
            
            # If there are exactly two differences, check if swapping makes them equal
            if len(diff) == 2:
                i, j = diff
                # Swap in sx and check if it matches sy
                swapped_sx = list(sx)
                swapped_sx[i], swapped_sx[j] = swapped_sx[j], swapped_sx[i]
                if ''.join(swapped_sx) == sy:
                    return True
            return False
        
        count = 0
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                if can_be_almost_equal(nums[i], nums[j]):
                    count += 1
        return count