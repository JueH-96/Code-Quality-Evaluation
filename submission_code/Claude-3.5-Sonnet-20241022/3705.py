class Solution:
    def largestInteger(self, nums: List[int], k: int) -> int:
        # Dictionary to store count of each number in each subarray
        num_counts = {}
        
        # Get all subarrays of size k and count occurrences of each number
        for i in range(len(nums) - k + 1):
            subarray = nums[i:i+k]
            for num in subarray:
                if num not in num_counts:
                    num_counts[num] = set()
                num_counts[num].add(i)
        
        # Find the largest number that appears in exactly one subarray
        max_almost_missing = -1
        for num, subarrays in num_counts.items():
            if len(subarrays) == 1:
                max_almost_missing = max(max_almost_missing, num)
                
        return max_almost_missing