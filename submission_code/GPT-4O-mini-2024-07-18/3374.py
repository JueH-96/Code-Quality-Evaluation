class Solution:
    def countAlternatingSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        
        for i in range(n):
            # Start a new subarray from index i
            length = 1
            
            for j in range(i + 1, n):
                if nums[j] != nums[j - 1]:
                    length += 1
                    count += length
                else:
                    break
        
        return count