class Solution:
    def maxFrequency(self, nums: List[int], k: int) -> int:
        n = len(nums)
        max_freq = 0
        
        # Count existing k's
        existing_k = nums.count(k)
        max_freq = existing_k
        
        # For each possible value, find the longest contiguous subarray
        # and see how many k's we can create
        for target_val in range(1, 51):  # Based on constraints
            if target_val == k:
                continue
                
            # Find all contiguous subarrays with value target_val
            i = 0
            while i < n:
                if nums[i] == target_val:
                    # Start of a potential subarray
                    j = i
                    while j < n and nums[j] == target_val:
                        j += 1
                    # Subarray from i to j-1 has all elements = target_val
                    # We can transform all of them to k
                    transformed = j - i
                    
                    # Count k's outside this range
                    outside_k = 0
                    for idx in range(i):
                        if nums[idx] == k:
                            outside_k += 1
                    for idx in range(j, n):
                        if nums[idx] == k:
                            outside_k += 1
                    
                    max_freq = max(max_freq, transformed + outside_k)
                    i = j
                else:
                    i += 1
        
        return max_freq