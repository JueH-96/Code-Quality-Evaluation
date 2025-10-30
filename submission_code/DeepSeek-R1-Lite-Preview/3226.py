class Solution:
    def numberGame(self, nums: List[int]) -> List[int]:
        # Sort the nums array in ascending order
        nums_sorted = sorted(nums)
        arr = []
        
        # Iterate through the sorted array two elements at a time
        for i in range(0, len(nums_sorted), 2):
            # Alice removes nums_sorted[i], Bob removes nums_sorted[i+1]
            # Bob appends first, then Alice appends
            arr.append(nums_sorted[i+1])
            arr.append(nums_sorted[i])
        
        return arr