class Solution:
    def beautifulSplits(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        
        # Count occurrences of each number in the array
        prefix_count = [0] * 51
        suffix_count = [0] * 51
        
        # Initialize suffix_count with the total counts of each number
        for num in nums:
            suffix_count[num] += 1
        
        # Iterate through the array to find beautiful splits
        for i in range(n - 1):
            # Update the prefix count
            prefix_count[nums[i]] += 1
            suffix_count[nums[i]] -= 1
            
            # Check for beautiful splits
            if i + 1 < n:
                # Check if nums1 is a prefix of nums2
                if prefix_count[nums[i]] > 0 and suffix_count[nums[i]] > 0:
                    count += 1
            
            # Check if nums2 is a prefix of nums3
            if i + 2 < n:
                if prefix_count[nums[i + 1]] > 0 and suffix_count[nums[i + 1]] > 0:
                    count += 1
        
        return count