class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        from collections import defaultdict
        
        # Dictionary to store the first occurrence of a value
        value_index = defaultdict(list)
        
        # Populate the dictionary with indices for each value
        for index, value in enumerate(nums):
            value_index[value].append(index)
        
        max_sum = 0
        found_good_subarray = False
        
        # Iterate through the array to find good subarrays
        for i in range(len(nums)):
            # Check for nums[i] + k in the dictionary
            if nums[i] + k in value_index:
                for j in value_index[nums[i] + k]:
                    if j > i:
                        subarray_sum = sum(nums[i:j+1])
                        if subarray_sum > max_sum:
                            max_sum = subarray_sum
                            found_good_subarray = True
            
            # Check for nums[i] - k in the dictionary
            if nums[i] - k in value_index:
                for j in value_index[nums[i] - k]:
                    if j > i:
                        subarray_sum = sum(nums[i:j+1])
                        if subarray_sum > max_sum:
                            max_sum = subarray_sum
                            found_good_subarray = True
        
        return max_sum if found_good_subarray else 0