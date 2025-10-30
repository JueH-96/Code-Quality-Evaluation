class Solution:
    def countArrays(self, original: List[int], bounds: List[List[int]]) -> int:
        n = len(original)
        
        lower_bound = float('-inf')
        upper_bound = float('inf')
        
        for i in range(n):
            u_i, v_i = bounds[i]
            diff = original[i] - original[0]
            
            # copy[0] must satisfy: u_i - diff <= copy[0] <= v_i - diff
            lower_bound = max(lower_bound, u_i - diff)
            upper_bound = min(upper_bound, v_i - diff)
        
        if lower_bound > upper_bound:
            return 0
        
        return upper_bound - lower_bound + 1