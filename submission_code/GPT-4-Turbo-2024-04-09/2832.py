class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        from collections import defaultdict
        
        def max_length_with_fixed_element(x):
            # This function calculates the maximum length of equal subarray
            # that can be formed with the element `x` using at most `k` deletions.
            max_len = 0
            current_count = 0
            left = 0
            deletions_used = 0
            
            for right in range(len(nums)):
                if nums[right] == x:
                    current_count += 1
                else:
                    deletions_used += 1
                
                # If we have used more deletions than allowed, shrink the window
                while deletions_used > k:
                    if nums[left] != x:
                        deletions_used -= 1
                    left += 1
                    current_count -= 1
                
                # Calculate the current length of the window
                current_len = right - left + 1
                max_len = max(max_len, current_len)
            
            return max_len
        
        # We can use a frequency map to determine which elements to consider
        frequency = defaultdict(int)
        for num in nums:
            frequency[num] += 1
        
        # We will try to find the longest equal subarray for each unique element
        max_equal_subarray_length = 0
        for num in frequency:
            max_equal_subarray_length = max(max_equal_subarray_length, max_length_with_fixed_element(num))
        
        return max_equal_subarray_length