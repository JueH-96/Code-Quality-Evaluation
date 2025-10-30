class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        diff = [0] * (n + 1)  # Difference array
        curr = 0             # Tracks how many times we've subtracted at the current index
        
        for i in range(n):
            curr += diff[i]          # Update the current subtraction count
            needed = nums[i] - curr  # How many more subtractions are needed to make nums[i] zero
            
            if needed < 0:
                # We've subtracted too much already; cannot fix this
                return False
            
            if needed > 0:
                # We need to subtract 'needed' times on subarray [i..i+k-1]
                if i + k > n:
                    # That subarray doesn't fit within the array bounds
                    return False
                curr += needed               # Increase current subtraction count
                diff[i + k] -= needed        # When we leave index i+k, reduce the subtraction count
        
        return True