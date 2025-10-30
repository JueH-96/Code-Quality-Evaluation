class Solution:
    def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        diff = [0] * n
        
        # To track distinct elements in prefix and suffix
        prefix_distinct_count = 0
        suffix_distinct_count = 0
        
        # To track occurrences of elements
        prefix_count = {}
        suffix_count = {}
        
        # Initialize suffix_count with all elements
        for num in nums:
            if num in suffix_count:
                suffix_count[num] += 1
            else:
                suffix_count[num] = 1
                suffix_distinct_count += 1
        
        # Iterate through each element and calculate prefix and suffix distinct counts
        for i in range(n):
            # Add current element to prefix count
            if nums[i] in prefix_count:
                prefix_count[nums[i]] += 1
            else:
                prefix_count[nums[i]] = 1
                prefix_distinct_count += 1
            
            # Remove current element from suffix count
            if suffix_count[nums[i]] == 1:
                del suffix_count[nums[i]]
                suffix_distinct_count -= 1
            else:
                suffix_count[nums[i]] -= 1
            
            # Calculate the difference
            diff[i] = prefix_distinct_count - suffix_distinct_count
        
        return diff