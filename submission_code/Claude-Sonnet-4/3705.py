class Solution:
    def largestInteger(self, nums: List[int], k: int) -> int:
        # Count how many subarrays of size k each number appears in
        count = {}
        
        # Generate all subarrays of size k
        for i in range(len(nums) - k + 1):
            subarray = nums[i:i+k]
            # Count each unique number in this subarray
            unique_nums = set(subarray)
            for num in unique_nums:
                count[num] = count.get(num, 0) + 1
        
        # Find the largest number that appears in exactly one subarray
        almost_missing = []
        for num, freq in count.items():
            if freq == 1:
                almost_missing.append(num)
        
        if not almost_missing:
            return -1
        
        return max(almost_missing)