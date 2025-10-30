class Solution:
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        n = len(nums)
        total = 0
        
        # Generate all subarrays
        for i in range(n):
            for j in range(i, n):
                # Get the current subarray
                subarray = nums[i:j+1]
                
                # Sort the subarray
                sorted_subarray = sorted(subarray)
                
                # Count imbalances (gaps > 1)
                imbalance = 0
                for k in range(len(sorted_subarray) - 1):
                    if sorted_subarray[k+1] - sorted_subarray[k] > 1:
                        imbalance += 1
                
                total += imbalance
        
        return total