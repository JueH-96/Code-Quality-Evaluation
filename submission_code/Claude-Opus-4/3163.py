class Solution:
    def sumCounts(self, nums: List[int]) -> int:
        n = len(nums)
        total_sum = 0
        
        # Iterate through all possible starting points
        for i in range(n):
            # Use a set to track distinct elements in current subarray
            distinct = set()
            
            # Extend the subarray from position i to j
            for j in range(i, n):
                # Add current element to the set
                distinct.add(nums[j])
                
                # Square the count of distinct elements and add to total
                total_sum += len(distinct) ** 2
        
        return total_sum