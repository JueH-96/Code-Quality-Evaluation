class Solution:
    def countArrays(self, original: List[int], bounds: List[List[int]]) -> int:
        n = len(original)
        
        # Compute differences from original[0]
        diff = [original[i] - original[0] for i in range(n)]
        
        # Find the range of valid x (copy[0])
        left = float('-inf')
        right = float('inf')
        
        for i in range(n):
            u_i, v_i = bounds[i]
            # u_i <= x + diff[i] <= v_i
            # u_i - diff[i] <= x <= v_i - diff[i]
            left = max(left, u_i - diff[i])
            right = min(right, v_i - diff[i])
        
        if left > right:
            return 0
        
        return right - left + 1