class Solution:
    def numberGame(self, nums: List[int]) -> List[int]:
        arr = []
        # We will repeat the process until nums is empty
        while nums:
            # Alice removes the minimum element
            x = min(nums)
            nums.remove(x)
            # Bob removes the new minimum element
            y = min(nums)
            nums.remove(y)
            
            # First Bob appends the element he removed (y), then Alice appends x
            arr.append(y)
            arr.append(x)
        
        return arr