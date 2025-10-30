class Solution:
    def sumCounts(self, nums: List[int]) -> int:
        n = len(nums)
        total_sum = 0
        # Enumerate all subarrays nums[i..j]
        for i in range(n):
            for j in range(i, n):
                # Calculate the number of distinct elements in this subarray
                distinct_count = len(set(nums[i:j+1]))
                # Add the square of the distinct count
                total_sum += distinct_count ** 2
        
        return total_sum