class Solution:
    def countArrays(self, original: List[int], bounds: List[List[int]]) -> int:
        n = len(original)
        low_bound = -10**18
        high_bound = 10**18
        
        for i in range(n):
            diff = original[i] - original[0]
            L_i = bounds[i][0] - diff
            R_i = bounds[i][1] - diff
            if L_i > low_bound:
                low_bound = L_i
            if R_i < high_bound:
                high_bound = R_i
        
        if low_bound > high_bound:
            return 0
        return high_bound - low_bound + 1