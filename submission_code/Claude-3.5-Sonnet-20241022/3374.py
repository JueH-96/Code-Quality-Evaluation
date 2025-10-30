class Solution:
    def countAlternatingSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        count = n # Single element subarrays are always alternating
        
        for i in range(n-1):
            # Start checking from length 2 subarrays
            if nums[i] != nums[i+1]:
                count += 1
                j = i + 2
                expected = nums[i]
                # Keep extending while pattern alternates
                while j < n and nums[j] == expected:
                    count += 1
                    j += 1
                    expected = 1 - expected
                    
        return count