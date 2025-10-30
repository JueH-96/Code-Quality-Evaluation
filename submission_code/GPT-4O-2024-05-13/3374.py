class Solution:
    def countAlternatingSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        
        # Function to count alternating subarrays ending at index i
        def count_alternating_ending_at(i):
            length = 1
            while i - length >= 0 and nums[i - length] != nums[i - length + 1]:
                length += 1
            return length
        
        for i in range(n):
            count += count_alternating_ending_at(i)
        
        return count