class Solution:
    def sumCounts(self, nums: List[int]) -> int:
        n = len(nums)
        total = 0
        
        # Generate all subarrays
        for i in range(n):
            distinct_set = set()
            for j in range(i, n):
                # Add current element to the set
                distinct_set.add(nums[j])
                # Count distinct elements and square it
                distinct_count = len(distinct_set)
                total += distinct_count * distinct_count
        
        return total