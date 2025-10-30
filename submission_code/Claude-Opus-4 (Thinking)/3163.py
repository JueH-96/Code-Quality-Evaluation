class Solution:
    def sumCounts(self, nums: List[int]) -> int:
        n = len(nums)
        total_sum = 0
        
        # Iterate through all possible starting positions
        for i in range(n):
            # Use a set to track distinct values as we extend the subarray
            distinct_values = set()
            
            # Extend the subarray from position i to j
            for j in range(i, n):
                # Add current element to our set of distinct values
                distinct_values.add(nums[j])
                
                # Get the count of distinct values
                distinct_count = len(distinct_values)
                
                # Add the square of distinct count to our total
                total_sum += distinct_count ** 2
        
        return total_sum