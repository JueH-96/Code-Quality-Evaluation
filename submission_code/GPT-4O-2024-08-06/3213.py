class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)
        max_num = max(nums)
        count = 0
        i = 0
        
        while i < n:
            if nums[i] == max_num:
                # Find the length of the contiguous segment of max_num
                j = i
                while j < n and nums[j] == max_num:
                    j += 1
                
                # Length of the segment
                length = j - i
                
                # If the segment length is at least k, calculate the number of valid subarrays
                if length >= k:
                    # Calculate the number of subarrays where max_num appears at least k times
                    for start in range(i, j - k + 1):
                        count += (j - (start + k) + 1)
                
                # Move i to the end of the current segment
                i = j
            else:
                i += 1
        
        return count