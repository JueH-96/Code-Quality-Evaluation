class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0

        # Initialize the count of valid subarrays
        count = 0

        # Iterate through each element in the array
        for i in range(n):
            max_val = nums[i]
            for j in range(i, n):
                max_val = max(max_val, nums[j])
                if nums[i] == max_val and nums[j] == max_val:
                    count += 1

        return count