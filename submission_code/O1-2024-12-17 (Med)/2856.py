class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        # First, determine how many distinct elements are in the entire array
        distinct_total = len(set(nums))
        
        n = len(nums)
        result = 0
        
        # For each starting index i, we expand j until we find a subarray that
        # contains all distinct elements (i.e., has 'distinct_total' distinct elements).
        # Once found, all extensions of that subarray to the right are also valid,
        # so we add (n - j) to the result and move on to the next i.
        
        for i in range(n):
            freq = {}
            distinct_count = 0
            for j in range(i, n):
                freq[nums[j]] = freq.get(nums[j], 0) + 1
                if freq[nums[j]] == 1:
                    # We have a new distinct element in the current window
                    distinct_count += 1
                
                # If the current window has all distinct elements,
                # all subarrays starting at i and ending at j or beyond are complete
                if distinct_count == distinct_total:
                    result += (n - j)
                    break
        
        return result