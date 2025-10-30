class Solution:
    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
        # We will maintain an array of colors, initially all 0 (uncolored).
        nums = [0]*n
        
        # This variable will keep track of how many adjacent pairs of the same (non-zero) color we currently have.
        pairs = 0
        
        # A function to check if index i and index i+1 form a colored pair.
        def is_pair(i):
            return 0 <= i < n-1 and nums[i] != 0 and nums[i] == nums[i+1]
        
        ans = []
        for idx, color in queries:
            # If the color is the same as current, no change in pairs
            if nums[idx] == color:
                ans.append(pairs)
                continue
            
            # Remove pairs involving idx (if any) before changing its color
            if is_pair(idx-1):
                pairs -= 1
            if is_pair(idx):
                pairs -= 1
            
            # Change the color
            nums[idx] = color
            
            # Add pairs involving idx (if any) after changing its color
            if is_pair(idx-1):
                pairs += 1
            if is_pair(idx):
                pairs += 1
            
            ans.append(pairs)
        
        return ans