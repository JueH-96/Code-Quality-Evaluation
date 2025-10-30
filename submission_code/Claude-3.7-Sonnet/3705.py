class Solution:
    def largestInteger(self, nums: List[int], k: int) -> int:
        from collections import defaultdict
        
        # Dictionary to track which subarrays each number appears in
        appearance_in_subarrays = defaultdict(set)
        
        # Generate all subarrays of size k
        for i in range(len(nums) - k + 1):
            for j in range(i, i+k):
                appearance_in_subarrays[nums[j]].add(i)
        
        # Find numbers that appear in exactly one subarray
        almost_missing = []
        for num, subarrays in appearance_in_subarrays.items():
            if len(subarrays) == 1:
                almost_missing.append(num)
        
        # Return the largest almost missing number, or -1 if none exist
        return max(almost_missing) if almost_missing else -1