class Solution:
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        n = len(nums)
        total = 0
        
        for i in range(n):
            for j in range(i, n):
                # Get subarray from i to j (inclusive)
                subarray = nums[i:j+1]
                # Sort it
                subarray.sort()
                # Count imbalances
                for k in range(len(subarray) - 1):
                    if subarray[k+1] - subarray[k] > 1:
                        total += 1
        
        return total