class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        from collections import defaultdict

        # Calculate the number of distinct elements in the whole array
        total_distinct = len(set(nums))
        
        # Initialize the count of complete subarrays
        complete_subarrays_count = 0
        
        # Iterate over each starting point of the subarray
        for start in range(len(nums)):
            # Use a dictionary to count the frequency of elements in the current subarray
            freq = defaultdict(int)
            distinct_count = 0
            
            # Iterate over each ending point of the subarray
            for end in range(start, len(nums)):
                # If the element is not in the current subarray, increase the distinct count
                if freq[nums[end]] == 0:
                    distinct_count += 1
                # Increase the frequency of the current element
                freq[nums[end]] += 1
                
                # If the number of distinct elements matches the total distinct elements
                if distinct_count == total_distinct:
                    # Increment the count of complete subarrays
                    complete_subarrays_count += 1
        
        return complete_subarrays_count